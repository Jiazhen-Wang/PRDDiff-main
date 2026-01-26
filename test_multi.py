import argparse
import logging
import os
import numpy as np
import torch
import logger as Logger
from models import create_model
import utils as util
from mri_dataset import create_dataloader, create_dataset
import csv


###############################################################
# 主函数
###############################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default='./configs/testdpm.yml', help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--low", type=int, default=40)
    parser.add_argument("--middle", type=int, default=200)
    parser.add_argument("--high", type=int, default=320)
    parser.add_argument("--use_y", default=True)
    parser.add_argument("--path", default='./savemodel/270000_RG.pth')
    parser.add_argument("--save_dir_mat", default='./result/')
    args = parser.parse_args()

    opt = Logger.parse(args.opt, args.low, args.high, args.use_y, args.path, is_train=False)

    opt = Logger.dict_to_nonedict(opt)
    time_l = int((320 - args.low) / 2)
    time_s1 = int((320 - args.middle) / 2)
    time_s = int((320 - args.high) / 2)
    seed = opt["train"]["manual_seed"]
    util.mkdirs(
        (
            path
            for key, path in opt["path"].items()
            if not key == "experiments_root"
               and "pretrain_model" not in key
               and "resume" not in key
        )
    )

    # cudnn加速训练
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 设置log文件记录实验
    util.setup_logger(
        "base",
        opt["path"]["log"],
        "test_" + opt["name"],
        level=logging.INFO,
        screen=False,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(Logger.dict2str(opt))

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt, phase)

        logger.info(
            "Number of test images in [{:s}]: {:d}".format(
                dataset_opt["name"], len(test_set)
            )
        )
        test_loaders.append(test_loader)
    opt['path']['pretrain_model'] = args.path


    # Create Model
    model = create_model(opt)
    device = model.device


    sde = util.SRDPM(max_sigma=opt["sde"]["max_sigma"], N=opt["sde"]["T"], noise_schedule=opt["sde"]["schedule"],
                     eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    avg_sde_plus_psnr_list = []
    avg_sde_plus_ssim_list = []
    avg_sde_plus_psnr_s_list = []
    avg_sde_plus_ssim_s_list = []
    avg_sde_plus_lpips_list = []
    avg_sde_plus_nrmse_list = []

    idx = 0
    headers = ['number', 'id', 'type', 'psnr', 'psnr_s', 'ssim', 'ssim_s', 'lpips', 'nrmse']
    result_path = '{}'.format(opt['path']['val_images'])
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if args.use_y:
        use_y = 'usey'
    else:
        use_y = 'noy'
    with open('result/{}_{}_{}_{}.csv'.format(args.low, args.high, args.path.split('/')[-1][:-7], use_y), 'w',
              encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for _, (val_data, id) in enumerate(test_loader):
            idx += 1

            result_path_idx = '{}/{}'.format(opt['path']['val_images'], idx)
            if not os.path.exists(result_path_idx):
                os.makedirs(result_path_idx)

            x0 = val_data
            ####Stage 1#######
            deg_xt, mask_l, y, condition, xs, mask_s1 = sde.get_deg_t_s(x0, time_l, time_s1)
            model.feed_data(deg_xt, x0=x0, condition=condition)
            model.test_sde_t_s(sde, mask=mask_l, y=y, condition=condition, use_inver_fp=args.use_y, time_t=time_l,
                               time_s=time_s1)
            visuals = model.get_current_visuals()


            ####Stage 2#######
            middle_sr = visuals["Re"]
            middle_sr = middle_sr.unsqueeze(0)

            deg_xt, mask_s1, y, condition, xs, mask_s = sde.get_deg_t_s_two(x0, middle_sr, time_s1, time_s)
            model.feed_data(deg_xt, x0=x0, condition=condition)
            model.test_sde_t_s(sde, mask=mask_s1, y=y, condition=condition, use_inver_fp=args.use_y, time_t=time_s1,
                               time_s=time_s)

            visuals1 = model.get_current_visuals()

            gt = util.tensor2mriimg(visuals["GT"].squeeze())
            de = util.tensor2mriimg(visuals["Input"].squeeze())
            de_inf = util.tensor2mriimg(y.squeeze())
            gt_img = np.abs(gt)
            de_img = np.abs(de)
            inf_img = np.abs(de_inf)

            re_sde_plus = util.tensor2mriimg(visuals1["Re"].squeeze())

            re_sde_plus_img = np.abs(re_sde_plus)
            sde_plus_psnr = util.cal_psnr(re_sde_plus, gt)
            sde_plus_ssim = util.cal_ssim(re_sde_plus_img, gt_img)
            sde_plus_lpips = util.cal_lpips(re_sde_plus_img, gt_img)
            sde_plus_nrmse = util.cal_nrmse(re_sde_plus_img, gt_img)
            ####   matching gt image at the resolution of time step s
            gt_s = util.get_xs_s_mean(x0, mask_s)
            ####   final resolution matching step to match SR image at the resolution of time step s
            re_s = util.get_xs_s_mean(visuals1["Re"].unsqueeze(0), mask_s)


            #####
            gt_s_img = util.tensor2mriimg(gt_s.squeeze())
            re_s_img = util.tensor2mriimg(re_s.squeeze())

            sde_plus_psnr_s = util.cal_psnr(re_s_img, gt_s_img)
            sde_plus_ssim_s = util.cal_ssim(re_s_img, gt_s_img)

            avg_sde_plus_psnr_list.append(sde_plus_psnr) ####for y_0 psnr
            avg_sde_plus_ssim_list.append(sde_plus_ssim) ####for y_0 ssim
            avg_sde_plus_lpips_list.append(sde_plus_lpips) ####for y_0 lpips
            avg_sde_plus_nrmse_list.append(sde_plus_nrmse) ####for y_0 nrmse
            avg_sde_plus_psnr_s_list.append(sde_plus_psnr_s) ####for y_s psnr
            avg_sde_plus_ssim_s_list.append(sde_plus_ssim_s) ####for y_s ssim



            # Save Images
            util.save_mriimg(re_sde_plus_img, '{}/sde_plus_{}.png'.format(result_path, idx))
            rows = [
                [str(_ + 1), str(id[0]), 'sde_plus', sde_plus_psnr, sde_plus_psnr_s, sde_plus_ssim, sde_plus_ssim_s,
                 sde_plus_lpips, sde_plus_nrmse]]
            writer.writerows(rows)

            logger.info(
                "# Num:{}   SDE # PSNR: {:.6f}  PSNRs: {:.6f} SSIM: {:.6f} ,SSIMs: {:.6f}  LPIPS: {:.6f} NRMSE : {:.6f} ".format(
                    str(id[0]), sde_plus_psnr, sde_plus_psnr_s, sde_plus_ssim, sde_plus_ssim_s, sde_plus_lpips,
                    sde_plus_nrmse))



        avg_sde_plus_psnr = np.mean(avg_sde_plus_psnr_list)
        avg_sde_plus_ssim = np.mean(avg_sde_plus_ssim_list)
        avg_sde_plus_lpips = np.mean(avg_sde_plus_lpips_list)
        avg_sde_plus_nrmse = np.mean(avg_sde_plus_nrmse_list)
        avg_sde_plus_psnr_s = np.mean(avg_sde_plus_psnr_s_list)
        avg_sde_plus_ssim_s = np.mean(avg_sde_plus_ssim_s_list)

        std_sde_plus_psnr = np.std(avg_sde_plus_psnr_list)
        std_sde_plus_ssim = np.std(avg_sde_plus_ssim_list)
        std_sde_plus_lpips = np.std(avg_sde_plus_lpips_list)
        std_sde_plus_nrmse = np.std(avg_sde_plus_nrmse_list)
        std_sde_plus_psnr_s = np.std(avg_sde_plus_psnr_s_list)
        std_sde_plus_ssim_s = np.std(avg_sde_plus_ssim_s_list)

        rows = [
            ['avg', 'avg', 'sde_plus', avg_sde_plus_psnr, avg_sde_plus_psnr_s, avg_sde_plus_ssim, avg_sde_plus_ssim_s,
             avg_sde_plus_lpips, avg_sde_plus_nrmse]]
        writer.writerows(rows)
        rows = [
            ['std', 'std', 'sde_plus', std_sde_plus_psnr, std_sde_plus_psnr_s, std_sde_plus_ssim, std_sde_plus_ssim_s,
             std_sde_plus_lpips, std_sde_plus_nrmse]]
        writer.writerows(rows)



    logger.info("----------------------------|AVG AVG |----------------------------------")
    logger.info(
        "# SDEAVG         # PSNR: {:.6f} , PSNRs: {:.6f}  SSIM: {:.6f} , SSIMs: {:.6f}  LPIPS: {:.6f}, NRMSE: {:.6f} ".format(
            avg_sde_plus_psnr, avg_sde_plus_psnr_s, avg_sde_plus_ssim, avg_sde_plus_ssim_s, avg_sde_plus_lpips,
            avg_sde_plus_nrmse))
    logger.info(
        "# SDESTD         # PSNR: {:.6f} , PSNRs: {:.6f}  SSIM: {:.6f} , SSIMs: {:.6f}  LPIPS: {:.6f}, NRMSE: {:.6f} ".format(
            std_sde_plus_psnr, std_sde_plus_psnr_s, std_sde_plus_ssim, std_sde_plus_ssim_s, std_sde_plus_lpips,
            std_sde_plus_nrmse))



if __name__ == "__main__":
    main()

#