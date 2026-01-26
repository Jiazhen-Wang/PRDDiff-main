import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class Score_Model(BaseModel):
    def __init__(self, opt):
        super(Score_Model, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
        train_opt = opt["train"]


        self.model = networks.get_net(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.model = DataParallel(self.model)


        self.load()


        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            

            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']



            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0



            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters(): 
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")


            self.op_loss = train_opt["op_loss"]
            self.time_small = train_opt["time_small"]
            

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

        if self.is_test_plus:
           print('test plus')

    def feed_data(self, xt, x0=None):
        self.xt = xt.to(self.device)
        self.input = self.xt

        if x0 is not None:
            self.gt = x0.to(self.device)  # gt
        if self.is_test_plus:
            print("test_plus")

    def optimize_parameters(self, step, timesteps, sde=None):
        
        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)


        x0_pred = sde.pred_fn(self.xt, timesteps.squeeze())
        
        if self.op_loss =="score":
            score_pred = sde.get_score_from_x0(x0_pred, self.xt,timesteps)
            score = sde.get_score_from_x0(self.gt, self.xt,timesteps)
            loss = self.weight * self.loss_fn(score_pred,score)

        elif self.op_loss == "sample":
            mask_t = sde.set_mask(timesteps)
            At_x = sde.get_xt_mean(x0_pred,mask_t)
            At_gt = sde.get_xt_mean(self.gt,mask_t)
            loss = self.weight * self.loss_fn(At_x,At_gt)
        elif self.op_loss == "sample+":
            timesteps_small =  torch.clamp(timesteps - self.time_small,min=0.0)
            mask_t = sde.set_mask(timesteps_small)
            At_x = sde.get_xt_mean(x0_pred,mask_t)
            At_gt = sde.get_xt_mean(self.gt,mask_t)
            loss = self.weight * self.loss_fn(At_x,At_gt)

        else:
             raise NotImplementedError("Not op_loss, default using score_loss.")
        loss.backward()
        self.optimizer.step()
        self.ema.update()


        self.log_dict["loss"] = loss.item()

    def test(self, sde=None,mask=None, save_states=False):

        self.model.eval()
        with torch.no_grad():
            if mask != None:

                self.output = sde.reverse_sde_plus(self.xt,mask,save_states=save_states)
            else:

                self.output = sde.reverse_sde(self.xt, save_states=save_states)

        self.model.train()


    def test_plus(self, sde=None, save_states=False):

        self.model.eval()
        
        with torch.no_grad():
            self.output = sde.reverse_sde_plus(self.xt, save_states=save_states)

        self.model.train()



    def get_current_log(self):
        return self.log_dict


    def get_current_visuals(self, need_GT=True):


        out_dict = OrderedDict()


        out_dict["Input"] = self.input.detach()[0].float().cpu()
        out_dict["Re"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.gt.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)


    def load(self):
        load_path_M = self.opt["path"]["pretrain_model"]
        if load_path_M is not None:
            logger.info("Loading model for  [{:s}] ...".format(load_path_M))
            self.load_network(load_path_M, self.model, self.opt["path"]["strict_load"])


    def save(self, iter_label,epoch):
        self.save_network(self.model, epoch, iter_label)
        self.save_network(self.ema.ema_model, "EMA", iter_label)


    def save_last(self, iter_label):
        self.save_network(self.model, 'NOEMA' ,iter_label)
        self.save_network(self.ema.ema_model, "EMA", iter_label)
