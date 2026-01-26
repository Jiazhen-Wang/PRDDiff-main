import argparse
import logging
import math
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logger as Logger
from models import create_model
import utils as util
from mri_dataset import create_dataloader, create_dataset
from mri_dataset.data_sampler import DistIterSampler

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
            mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


###############################################################
# main
###############################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default='configs/traindpm.yml', help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    opt = Logger.parse(args.opt, is_train=True)
    opt = Logger.dict_to_nonedict(opt)

    seed = opt["train"]["manual_seed"]

    ####################
    # Distributed Training Config
    ####################

    if args.launcher == "none":
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()

        world_size = (
            torch.distributed.get_world_size()
        )

        rank = torch.distributed.get_rank()

    ########################################
    # CuDNN Training Acceleration
    ########################################
    torch.backends.cudnn.benchmark = True

    ####################
    # Check and Load Previous Training State (if exists)
    ####################
    if opt["path"].get("resume_state", None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        Logger.check_resume(opt, resume_state["iter"])
    else:
        resume_state = None

    ############################################
    # Create Experiment Log File
    ############################################

    if rank <= 0:
        if resume_state is None:
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                       and "pretrain_model" not in key
                       and "resume" not in key
                )
            )

            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(Logger.dict2str(opt))

        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                import tensorboardX
                from tensorboardX import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    ##########################################
    # Create Training and Validation Datasets
    ##########################################

    dataset_ratio = 10

    for phase, dataset_opt in opt["datasets"].items():

        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))

            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None

            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )

        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, phase)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))

    assert train_loader is not None
    assert val_loader is not None

    ####################
    # Create Model
    ####################
    model = create_model(opt)
    device = model.device
    logger.info(model.model)

    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)

    else:
        current_step = 0
        start_epoch = 0

    if opt["name"] == "sr-sde":
        sde = util.SRSDE(max_sigma=opt["sde"]["max_sigma"], N=opt["sde"]["T"], noise_schedule=opt["sde"]["schedule"],
                         eps=opt["sde"]["eps"], device=device)
    elif opt["name"] == "sr-dpm":
        sde = util.SRDPM(max_sigma=opt["sde"]["max_sigma"], N=opt["sde"]["T"], noise_schedule=opt["sde"]["schedule"],
                         eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)



    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_sde_psnr = 0.0
    best_sde_plus_psnr = 0.0

    best_sde_psnr_iter = 0
    best_sde_plus_psnr_iter = 0

    error = mp.Value('b', False)

    ###################
    # Training
    ###################

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            # Get Training Data
            data, id = train_data
            x0 = data

            # Return Time step and Noisy degraded input
            timesteps, xt, condition, time_s, xs, res, mask_s = sde.generate_random_degenerate_gu(x0=x0, is_con=True)

            #  Feed Data to Model
            model.feed_data(xt, x0, xs, condition=condition, res=res, mask_s=mask_s)  # xt, x0

            # Model Optimization
            model.optimize_parameters(current_step, timesteps, res, sde)

            # Update Learning Rate
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )


            if current_step % opt["logger"]["print_freq"] == 0:

                logs = model.get_current_log()

                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )

                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)

                    # tensorboard
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:

                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            ###################
            # Validation
            ###################
            if opt["train"]["need_vel"]:
                if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                    result_path = '{}/{}'.format(opt['path']['val_images'], epoch)
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)

                    avg_inf_psnr = 0.0
                    avg_inf_ssim = 0.0
                    avg_y_psnr = 0.0
                    avg_y_ssim = 0.0

                    avg_base_psnr = 0.0
                    avg_base_ssim = 0.0
                    avg_base_plus_psnr = 0.0
                    avg_base_plus_ssim = 0.0

                    avg_sde_psnr = 0.0
                    avg_sde_ssim = 0.0
                    avg_sde_plus_psnr = 0.0
                    avg_sde_plus_ssim = 0.0

                    avg_mix_psnr = 0.0
                    avg_mix_ssim = 0.0

                    idx = 0
                    #### for 40*40 ------> 320*320
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        x0, id = val_data
                        deg_xt, mask_t, y, condition = sde.get_deg_t(x0)
                        model.feed_data(deg_xt, x0=x0, condition=condition)
                        model.test_sde(sde, mask=mask_t, y=y, condition=condition, use_inver_fp=True)
                        visuals = model.get_current_visuals()

                        gt = util.tensor2mriimg(visuals["GT"].squeeze())
                        de = util.tensor2mriimg(visuals["Input"].squeeze())
                        de_inf = util.tensor2mriimg(y.squeeze())
                        de_inf_img = np.abs(de_inf)
                        gt_img = np.abs(gt)
                        de_img = np.abs(de)

                        re_sde = util.tensor2mriimg(visuals["Re"].squeeze())
                        re_sde_img = np.abs(re_sde)
                        # Calculate PSNR and SSIM
                        avg_y_psnr += util.cal_psnr(de, gt)
                        avg_y_ssim += util.cal_ssim(de_img, gt_img)

                        avg_inf_psnr += util.cal_psnr(de_inf, gt)
                        avg_inf_ssim += util.cal_ssim(de_inf_img, gt_img)

                        avg_sde_psnr += util.cal_psnr(re_sde, gt)
                        avg_sde_ssim += util.cal_ssim(re_sde_img, gt_img)

                        # Save Images
                        util.save_mriimg(gt_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx))
                        util.save_mriimg(de_img, '{}/{}_{}_de.png'.format(result_path, current_step, idx))
                        util.save_mriimg(re_sde_img, '{}/{}_{}_re_sde.png'.format(result_path, current_step, idx))
                        util.save_mriimg(de_inf_img, '{}/{}_{}_de_inf.png'.format(result_path, current_step, idx))

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.expand_dims(np.concatenate(
                                (de_img, re_sde_img, gt_img), axis=1), axis=0),
                            idx)

                    avg_sde_psnr = avg_sde_psnr / idx
                    avg_sde_ssim = avg_sde_ssim / idx


                    if avg_sde_psnr > best_sde_psnr:
                        best_sde_psnr = avg_sde_psnr
                        best_sde_psnr_iter = current_step


                    logger.info(
                        "----------------------------| epoch:{:3d}  Iter {} |----------------------------------".format(
                            epoch, current_step))
                    logger.info("# Validation # SDE  PSNR:           {:.6f}".format(avg_sde_psnr))
                    logger.info("# Validation # SDE  SSIM:           {:.6f}".format(avg_sde_ssim))
                    logger.info("# Validation # SDE  Best PSNR:      {:.6f}| Iter:      {}".format(best_sde_psnr,
                                                                                                   best_sde_psnr_iter))
                    logger.info("# Validation # SDE  Best PSNR_PLUS: {:.6f}| Iter:      {}".format(best_sde_plus_psnr,
                                                                                                   best_sde_plus_psnr_iter))


                    logger_val = logging.getLogger("val")
                    # PSNR
                    logger_val.info(
                        "----------------------------| epoch:{:3d}  Iter {} |----------------------------------".format(
                            epoch, current_step))
                    logger_val.info(
                        "<base_psnr: {:.6f}  sde_psnr: {:.6f}  base_plus_psnr: {:.6f}  sde_plus_psnr : {:.6f} mix_psnr : {:.6f}".format(
                            avg_base_psnr, avg_sde_psnr, avg_base_plus_psnr, avg_sde_plus_psnr, avg_mix_psnr
                        )
                    )
                    logger_val.info(
                        "<base_ssim: {:.6f}   sde_ssim: {:.6f}   base_plus_ssim: {:.6f}   sde_plus_ssim : {:.6f} mix_ssim : {:.6f}".format(
                            avg_base_ssim, avg_sde_ssim, avg_base_plus_ssim, avg_sde_plus_ssim, avg_mix_ssim
                        )
                    )
                    print("<epoch:{:3d}, iter:{:8,d}, base_psnr     : {:.6f}".format(
                        epoch, current_step, avg_base_psnr
                    ))

                    print("<epoch:{:3d}, iter:{:8,d}, base_plus_psnr: {:.6f}".format(
                        epoch, current_step, avg_base_plus_psnr
                    ))

                    print("<epoch:{:3d}, iter:{:8,d}, sde_psnr      : {:.6f}".format(
                        epoch, current_step, avg_sde_psnr
                    ))

                    print("<epoch:{:3d}, iter:{:8,d}, sde_plus_psnr : {:.6f}".format(
                        epoch, current_step, avg_sde_plus_psnr
                    ))

                    # SSIM

                    print("<epoch:{:3d}, iter:{:8,d}, base_ssim: {:.6f}".format(
                        epoch, current_step, avg_base_ssim
                    ))

                    print("<epoch:{:3d}, iter:{:8,d}, base_plus_ssim: {:.6f}".format(
                        epoch, current_step, avg_base_plus_ssim
                    ))

                    print("<epoch:{:3d}, iter:{:8,d}, sde_ssim: {:.6f}".format(
                        epoch, current_step, avg_sde_ssim
                    ))

                    print("<epoch:{:3d}, iter:{:8,d}, sde_plus_ssim: {:.6f}".format(
                        epoch, current_step, avg_sde_plus_ssim
                    ))

                    # tensorboard
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        tb_logger.add_scalar("sde psnr", avg_sde_psnr, current_step)
                        tb_logger.add_scalar("sde ssim", avg_sde_ssim, current_step)
                        tb_logger.add_scalar("sde plus_psnr", avg_sde_plus_psnr, current_step)
                        tb_logger.add_scalar("sde plus_ssim", avg_sde_plus_ssim, current_step)

                        tb_logger.add_scalars("psnr",
                                              {'sde': avg_sde_psnr, 'sde_plus': avg_sde_plus_psnr, 'inf': avg_inf_psnr,
                                               'de': avg_y_psnr}, current_step)
                        tb_logger.add_scalars("ssim",
                                              {'sde': avg_sde_ssim, 'sde_plus': avg_sde_plus_ssim, 'inf': avg_inf_ssim,
                                               'de': avg_y_ssim}, current_step)

            if error.value:
                sys.exit(0)

            #####################################
            # Save Model and Training State
            #####################################

            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step, epoch)
                    if current_step % (opt["logger"]["save_checkpoint_freq"] * 20) == 0:
                        model.save_training_state(epoch, current_step)


    if rank <= 0:
        logger.info("Saving the final model.")
        model.save_last("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()


