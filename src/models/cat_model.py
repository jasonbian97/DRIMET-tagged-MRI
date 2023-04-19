from typing import Any, List

import matplotlib.pyplot as plt
import path
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanSquaredError
from src.models.components.ScalingAndSquaring import VecInt
from src.models.components.SpatialWarp import SpatialWarp
import torch.nn as nn
import torch.nn.functional as F
from src.datamodules.components.datautils import compute_det_auc, compute_det_auc_ten, compute_neg_det_frac_ten
from src.models.components.SpatialWarp import scale_flow
import pandas as pd
import numpy as np


class CatModel(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            warper: torch.nn.Module,
            val_warper: torch.nn.Module,
            loss: torch.nn.Module,
            name: str = "anonymous",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "warper", "val_warper", "loss"])

        # take fixed and moving images, return flow maps. Mainly backbone network
        self.net = net

        # loss function
        self.criterion = loss

        # Spatial warper. Can be image warper or phase warper.
        self.warper = warper

        # this phase warper is only used in validation/test step
        self.val_warper = val_warper

        # scaling and squaring
        self.VecInt = VecInt(nsteps=7)
        # VoxelMorph performs scaling and squaring on downsampled size of flowmap to reduce computation cost.

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = MeanSquaredError()
        self.val_acc = MeanSquaredError()
        self.test_acc = MeanSquaredError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_sim = MeanMetric()
        self.val_loss_sm = MeanMetric()
        self.val_loss_in = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_det_auc = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MinMetric()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(x, y)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        # TODO: add senatio when
        x, y = batch
        C = x.shape[1]
        if C == 4:
            img1 = x[:, 0:3, :, :, :]
            img2 = y[:, 0:3, :, :, :]
            mag1 = x[:, 3:4, :, :, :]
            mag2 = y[:, 3:4, :, :, :]

        elif C == 7:  # sin_cos form input
            img1 = x[:, 0:6, :, :, :]
            img2 = y[:, 0:6, :, :, :]
            mag1 = x[:, 6:7, :, :, :]
            mag2 = y[:, 6:7, :, :, :]
        else:
            raise ValueError("Input channel should be 3+1 or 2+2+2+1")

        flow = self.forward(img1, img2)
        flow_deff = self.VecInt(flow)
        img2_ = self.warper(img2, flow_deff)

        loss, l_sim, l_sm, l_in = self.criterion(img1, img2_, flow, mag1)

        return loss, img1, img2, img2_, flow_deff, {"l_sim": l_sim, "l_sm": l_sm, "l_in": l_in, "mag1": mag1,
                                                    "mag2": mag2}

    def training_step(self, batch: Any, batch_idx: int):
        loss, img1, img2, img2_, flow_deff, other = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(img1, img2_)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "flow": flow_deff}

    # def training_epoch_end(self, outputs: List[Any]):
    #     #Overwriting this funciton will cause gpu&cpu memory leak!!!
    #     # `outputs` is a list of dicts returned from `training_step()`
    #     pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, img1, img2, _, flow_deff, other = self.step(batch)
        # img2_should be warped by phase warper for mse computing, not by image warper
        img2_ = self.val_warper(img2, flow_deff)
        B, C, H, W, D = batch[0].shape
        if C == 4:
            mag1 = batch[0][:, 3:4, :, :, :]
        elif C == 7:
            mag1 = batch[0][:, 6:7, :, :, :]

        scaled_flow = scale_flow(flow_deff, (H / 2., W / 2., D / 2.))  # scale the magnitude of normalized flow (-1,1)
        det_auc = compute_det_auc_ten(scaled_flow, weight=mag1)

        # update and log metrics
        self.val_loss(loss)
        self.val_det_auc(det_auc)
        self.val_loss_sim(other["l_sim"])
        self.val_loss_sm(other["l_sm"])
        self.val_loss_in(other["l_in"])
        self.val_acc(img1, img2_)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/det_auc", self.val_det_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/l_sim", self.val_loss_sim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/l_sm", self.val_loss_sm, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/l_in", self.val_loss_in, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        # vis_check
        # flow is can be interpreted as a color (rgb) image. e.g., -x direction (negative value in 1st-dim) is in Cyan.
        if batch_idx < 3:
            if "brain" in self.logger.experiment.tags:
                self.logger.log_image(key="warped_img_flow",
                                      images=[img1[0, 0, :, :, 64], img1[0, 1, :, :, 64],
                                              img2[0, 0, :, :, 64], img2[0, 1, :, :, 64],
                                              img2_[0, 0, :, :, 64], img2_[0, 1, :, :, 64],
                                              flow_deff[0, :, :, :, 64]],
                                      caption=["img1_c1", "img1_c2", "img2_c1", "img2_c2", "warped_c1", "warped_c2",
                                               "flow"],
                                      step=self.global_step)
            else:
                self.logger.log_image(key="warped_img_flow",
                                      images=[img1[0, 0, 32, :, :], img1[0, 1, 32, :, :],
                                              img2[0, 0, 32, :, :], img2[0, 1, 32, :, :],
                                              img2_[0, 0, 32, :, :], img2_[0, 1, 32, :, :],
                                              flow_deff[0, :, 32, :, :]],
                                      caption=["img1_c1", "img1_c2", "img2_c1", "img2_c2", "warped_c1", "warped_c2",
                                               "flow"],
                                      step=self.global_step)

        return {"loss": loss}  # do not return large thing here, otherwise it will quickly fill up gpu memory!!!

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/mse_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, img1, img2, _, flow_deff, other = self.step(batch)

        img2_ = self.val_warper(img2, flow_deff)
        mag1 = other["mag1"]

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(img1, img2_)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        metr_mse = F.mse_loss(img1, img2_)
        H, W, D = flow_deff.shape[2:]
        scaled_flow = scale_flow(flow_deff, (H / 2., W / 2., D / 2.))

        auc = compute_det_auc_ten(scaled_flow, weight=mag1)
        negdet_frac = compute_neg_det_frac_ten(scaled_flow, weight=None)

        # save scaled_flow to disk
        save_dir = path.Path(f"data/munge/R_eval_{self.hparams.name}")
        save_dir.makedirs_p()
        save_dic = {"img1": img1.cpu().numpy(), "img2": img2.cpu().numpy(), "img2_": img2_.cpu().numpy(),
                    "flow": scaled_flow.cpu().numpy(),
                    "mse": metr_mse.item(), "det_auc": auc, "negdet_frac": negdet_frac
                    }
        np.savez(save_dir / f"{batch_idx:04d}.npz", **save_dic)

        return {"loss": loss, "flow": flow_deff, "mse": metr_mse.item(), "det_auc": auc, "negdet_frac": negdet_frac}

    def test_epoch_end(self, outputs: List[Any]):

        rdict = {"mse": [], "det_auc": []}
        rdict["mse"] = [x["mse"] for x in outputs]
        rdict["det_auc"] = [x["det_auc"] for x in outputs]
        rdict["negdet_frac"] = [x["negdet_frac"] for x in outputs]

        df = pd.DataFrame(rdict)
        csv_path = f"data/munge/R_eval_{self.hparams.name}.csv"
        df.to_csv(csv_path, index_label="index")
        print("Save test results to ", csv_path)
        return rdict

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "cat.yaml")
    _ = hydra.utils.instantiate(cfg)
