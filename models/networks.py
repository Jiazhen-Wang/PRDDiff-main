import logging

logger = logging.getLogger("base")

# 得到网络结构
def get_net(opt):
    opt_net = opt["net"]
    
    model_name = opt_net["net_model"]
    if model_name == 'ConditionalUNet' or model_name == 'ConditionalNAFNet':
        from models import dpm as M
        setting = opt_net["dpm_set"]
        netG = getattr(M, model_name)(**setting) # 这里用getattr是因为有两个ddpm的net需要根据mode_name选择一下
    elif model_name == 'NCSN++':
        from models.ncsnplus.ncsnpp import NCSNpp as M
        from models import score_set
        config = score_set.get_config()
        netG = M(config)
    elif model_name == "GDPM":
        from models.gdpm import create_model
        setting = opt_net["gdpm"]
        netG = create_model(**setting)
    else:
        NotImplementedError("Model [{:s}] not recognized.".format(model_name))
    return netG



