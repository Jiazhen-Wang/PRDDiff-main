import logging

logger = logging.getLogger("base")

def create_model(opt):
    model = opt["model"]
    if model == "recon":
        from .score_model import Score_Model as M
    elif model =="recondpm":
        from . ddp_model import Diffusion_Model as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m

