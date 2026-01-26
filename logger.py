import logging
import os
import os.path as osp
import sys
import math

import yaml

try:
    #sys.path.append("../../") 
    from utils.file_util import OrderedYaml
except ImportError:
    pass

Loader, Dumper = OrderedYaml()


#设置一些实验参数
def parse(opt_path,low=40,high=320 ,use_y=True,path_name=None,is_train=True):
    with open(opt_path, mode="r") as f:
        opt = yaml.load(f, Loader=Loader)
    
    # 配置Cuda
    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    # 判断是训练还是测试
    opt["is_train"] = is_train

    

    # 数据集的一些参数设置，包括数据格式 路径 等等
    for phase, dataset in opt["datasets"].items():
        phase = phase.split("_")[0]
        print(dataset)
        dataset["phase"] = phase
        
        is_lmdb = False

        if dataset.get("dataroot", None) is not None:
            dataset["dataroot"] = osp.expanduser(dataset["dataroot"])
            if dataset["dataroot"].endswith("lmdb"): # 判断是否一lmdb结尾
                is_lmdb = True
       
        dataset["data_type"] = "lmdb" if is_lmdb else "mat"


    # 关于实验的一些路径
    for key, path in opt["path"].items():
        if path and key in opt["path"] and key != "strict_load":
            opt["path"][key] = osp.expanduser(path)
    opt["path"]["root"] = osp.abspath(
        osp.join(__file__, osp.pardir) #有结构osp.pardir就往上跳出几个层级
    )
    path = osp.abspath(__file__)
    config_dir = path.split("/")[-2]

    # 因为训练需要好多文件夹，所以在这里设置了一些
    if is_train:
        experiments_root = osp.join(
            opt["path"]["root"], "experiments", config_dir, opt["name"]
        )
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["training_state"] = osp.join(experiments_root, "training_state")
        opt["path"]["log"] = experiments_root
        opt["path"]["val_images"] = osp.join(experiments_root, "val_images")

        # 如果debug的话，我们可以减少验证频率和输出频率，这样可以快速验证整个代码项目是否有问题
        if "debug" in opt["name"]:
            opt["train"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    # 测试
    else: 
        # results_root = osp.join(opt["path"]["root"], "results", config_dir)
        # opt["path"]["results_root"] = osp.join(results_root, opt["name"])
        # opt["path"]["log"] = osp.join(results_root, opt["name"])
        if use_y:
            experiments_root = osp.join(
                opt["path"]["root"], "experiments", config_dir, opt["name"],str(low),str(high),'use_y',path_name.split('/')[-1][:-7]
            )
        else:
            experiments_root = osp.join(
                opt["path"]["root"], "experiments", config_dir, opt["name"],str(low),str(high),'no_y',path_name.split('/')[-1][:-7]
            )
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["training_state"] = osp.join(experiments_root, "training_state")
        opt["path"]["log"] = experiments_root
        opt["path"]["val_images"] = osp.join(experiments_root, "val_images")


    return opt


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# 转换配置用的
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter):
    """检查与训练好的模型或者训练到一半的模型"""
    logger = logging.getLogger("base")

    # get函数获取对应键位的值 如果没有则赋为None
    
    # resume state 如果为True则 用一下与训练好的模型，如果为fasle则不用，下面的都不在记录了
    # 如果用了，也就是 pretrain_model 会被赋值与训练好的模型的路径
    if opt["path"]["resume_state"]:
        if (
            opt["path"].get("pretrain_model", None) is not None
        ):
            logger.warning(
                "pretrain_model path will be ignored when resuming training."
            )

        opt["path"]["pretrain_model"] = osp.join(
            opt["path"]["models"], "{}_G.pth".format(resume_iter)
        )
        logger.info("Set [pretrain_model] to " + opt["path"]["pretrain_model"])