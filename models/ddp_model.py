import logging
from collections import OrderedDict
import os
import numpy as np

import fastmri
import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class Diffusion_Model(BaseModel):
    def __init__(self, opt):
        super(Diffusion_Model, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
        train_opt = opt["train"]

        # define network
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
            self.reg_weight = opt['train']['reg_weight']
            self.skip_weight = opt['train']['skip_weight']
            self.distill_weight = opt['train']['distill_weight']
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
            

            self.log_dict = OrderedDict()
        

        if self.is_test_plus:
           print('test plus')

    def feed_data(self, xt, x0, xs=None, condition=None, res=None, mask_s=None):
        xt =  xt.to(self.device)
        condition = condition.to(self.device)
        xt = xt.float()
        self.xt =xt
        self.input = self.xt
        self.condition = condition
        self.xs = xs
        self.res = res
        if mask_s is not None:
            self.mask_s = mask_s.to(self.device)
        if x0 is not None:
            x0 = x0.to(self.device)
            x0 = x0.float()
            self.gt = x0  # gt
        """ if self.is_test_plus:
            print("test_plus") """

    def optimize_parameters(self, step, timesteps, res, dpm=None):
        
        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)
        timestepsnet = timesteps.float()

        model_input = torch.cat([self.xt],dim=1)
        pred = dpm.pred_fn(model_input, timestepsnet, res)
        
        if self.op_loss =="mean":
            mask_t = dpm.set_mask(timesteps)
            At_gt = dpm.get_xt_mean(self.gt,mask_t)
            loss = self.weight * self.loss_fn(pred,At_gt)
        elif self.op_loss == "pred_xs":
            pred_xs = dpm.process_x0(pred)
            loss = self.weight * self.loss_fn(pred_xs,self.xs)
        elif self.op_loss == "pred_x0":
            pred_x0 = dpm.process_x0(pred)
            loss = self.weight * self.loss_fn(pred_x0,self.gt)
        elif self.op_loss == "pred_x0+":
            pred_x0 = dpm.process_x0(pred)
            timesteps_small =  torch.clamp(timesteps - 1,min=0.0)
            timesteps_small = timesteps_small.long()
            mask_t_prev = dpm.set_mask(timesteps_small)

            At_x = dpm.get_xt_mean_k(pred_x0,mask_t_prev)
            At_gt = dpm.get_xt_mean_k(self.gt,mask_t_prev)

            loss_k = self.weight * self.loss_fn(At_x,At_gt)
            loss = loss_k
        elif self.op_loss == 'sec':
            pred_xs = dpm.process_x0(pred)
            loss_i = self.weight * self.loss_fn(pred_xs,self.xs)
            pred_xs_k  = fastmri.fft2c(pred_xs.permute(0,2,3,1))
            xs_k = fastmri.fft2c(self.xs.permute(0,2,3,1))
            loss_k = self.loss_fn(pred_xs_k*self.mask_s,xs_k*self.mask_s)
            loss = loss_i + loss_k
        
        else:
             raise NotImplementedError("Not op_loss, default using score_loss.")

        loss.backward()
        self.optimizer.step()

        self.log_dict["loss"] = loss.item()
        self.log_dict["loss_k"] = loss_k.item()
        self.log_dict["loss_i"] = loss_i.item()

    def test_base(self, sde=None,mask=None,y=None, save_states=False,save_dir=None):

        self.model.eval()
        with torch.no_grad():
            if mask != None:
                print('test_base_plus')
                self.output = sde.reverse_re_plus(self.xt,y,mask,save_states=save_states,save_dir=save_dir)
            else:
                print('test_base')
                self.output = sde.reverse_re(self.xt, save_states=save_states,save_dir=save_dir)

        self.model.train()


    def test_sde(self, sde=None, y=None,condition=None,mask=None,N=-1,save_states=False,save_dir=None,use_inver_fp=False,use_tv=False):
        self.model.eval()
        with torch.no_grad():
            print('test_sde_plus')
            self.output = sde.reverse_sde_plus_add(self.xt,y,condition,mask,N=N, save_states=save_states, save_dir='sde_state',use_inver_fp=use_inver_fp,use_tv=use_tv)

        self.model.train()

    def test_sde_t_s(self, sde=None, y=None,condition=None,mask=None,N=-1,save_states=False,save_dir=None,use_inver_fp=False,use_tv=False,time_t=None,time_s=None):
        self.model.eval()
        with torch.no_grad():
            print('test_sde_plus')
            _, self.output = sde.reverse_sde_plus_add_t_s(self.xt,y,condition,mask,N=N, save_states=save_states, save_dir='sde_state',use_inver_fp=use_inver_fp,use_tv=use_tv,time_t=time_t,time_s=time_s)

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
        self.save_network(self.model, "RG", iter_label)


    def save_last(self, iter_label):
        self.save_network(self.model, 'NOEMA' ,iter_label)

