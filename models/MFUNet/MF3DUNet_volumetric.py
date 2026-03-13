"""Motion Field U-Net (MF-U-net) model definition with definitions of custom loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms.functional as TF

from modelcomponents import RainNet3DNoNorm as RN

import matplotlib.pyplot as plt
from pysteps.visualization import plot_precip_field, quiver
from pysteps.verification import fss

class MF3DUNetVolumetric(pl.LightningModule):
    """Model for the Motion Field U-Net (MF-U-net) neural network."""

    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()

        self.personal_device = torch.device(config.train_params.device)
        self.network = RN(**config.model.rainnet)

        self.loss_ignore_value = config.model.loss.loss_ignore_value
        if config.model.loss.name == "mse":
            self.criterion = CustomMSELoss(self.loss_ignore_value)
        else:
            raise NotImplementedError(f"Loss {config.model.loss.name} not implemented!")

        self.phys_loss = PhysLoss()
        self.beta = config.model.phys_loss.beta

        self.ground_loss_weight = config.model.ground_loss.weight
        self.cmax_loss_weight = config.model.cmax_loss.weight

        # on which leadtime to train the NN on?
        self.train_leadtimes = config.model.train_leadtimes
        self.verif_leadtimes = config.train_params.verif_leadtimes
        # How many leadtimes to predict
        self.predict_leadtimes = config.prediction.predict_leadtimes

        # optimization parameters
        self.lr = float(config.model.lr)
        self.lr_sch_params = config.train_params.lr_scheduler
        self.automatic_optimization = False

    def forward(self, x):
        mf = self.network(x)
        mf[:,0] = 0 # Set the vertical motion field to zero
        leadtimes = self.verif_leadtimes
        if self.trainer.training:
            leadtimes = self.train_leadtimes
        elif self.trainer.testing or self.trainer.predicting:
            leadtimes = self.predict_leadtimes
        return self._extrapolate_3d(leadtimes, x[:,-1:], mf, self.trainer.state.stage), mf

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_sch_params.name is None:
            return optimizer
        elif self.lr_sch_params.name == "reduce_lr_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, **self.lr_sch_params.kwargs
            )
            return [optimizer], [lr_scheduler]
        elif self.lr_sch_params.name == "ExponentialLR":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, **self.lr_sch_params.kwargs
            )
            return [optimizer], [lr_scheduler]
        else:
            raise NotImplementedError("Lr scheduler not defined.")

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        x, y, _ = batch
        x = x[:,:,0:1].squeeze(2).float()
        y = y[:,:,0:1].squeeze(2).float()

        x[torch.isnan(x)] = self.loss_ignore_value
        y[torch.isnan(y)] = self.loss_ignore_value

        y_hat, mf = self(x)

        loss = 0
        cmax_loss = 0
        phys_loss = 0

        sequence = torch.cat([x, y], dim=1).squeeze(1)
        for i in range(8):
            x_pool = F.avg_pool3d(x.squeeze(1), kernel_size=(1, 2**i, 2**i), stride=(1, 2**i, 2**i))
            y_pool = F.avg_pool3d(y.squeeze(1), kernel_size=(1, 2**i, 2**i), stride=(1, 2**i, 2**i))
            mf_pool = F.avg_pool3d(mf, kernel_size=(1, 2**i, 2**i), stride=(1, 2**i, 2**i))
            mf_pool[:,1] = mf_pool[:,1] / (2**i)
            mf_pool[:,2] = mf_pool[:,2] / (2**i)

            sequence = torch.cat([x_pool, y_pool], dim=1)
            for i in range(sequence.shape[1]-1):
                extrapolated = self._extrapolate_3d(1, sequence[:,i:i+1], mf_pool, self.trainer.state.stage)
                loss += self.criterion(extrapolated, sequence[:,i+1:i+2])
                cmax_loss += self.criterion(torch.max(extrapolated, 2).values, torch.max(sequence[:,i+1:i+2], 2).values)

            phys_loss = phys_loss + self.phys_loss(mf_pool)
        
        loss = loss / (sequence.shape[1] - 1)
        loss = loss / 8

        cmax_loss = cmax_loss / (sequence.shape[1] - 1)
        cmax_loss = cmax_loss / 8

        phys_loss = phys_loss / 8

        if self.ground_loss_weight > 0:
            neg_mask = mf[:,0,0] < 0
            ground_loss = torch.sum(mf[:,0,0][neg_mask] ** 2) / (mf.shape[0] * mf.shape[-2] * mf.shape[-1])
            self.log("train_ground_loss", ground_loss.detach())
            ground_loss = ground_loss * self.ground_loss_weight
            loss = loss + ground_loss

        cmax_extra_loss = self.criterion(torch.max(y_hat, 2).values, torch.max(y, 2).values)
        
        self.manual_backward(loss + (phys_loss * self.beta) + (cmax_loss * self.cmax_loss_weight))

        self.log("train_crit_loss", loss.detach())
        self.log("train_phys_loss", phys_loss.detach())
        self.log("train_cmax_loss", cmax_loss.detach())
        self.log("train_extra_loss", cmax_extra_loss.detach())

        opt.step()
        opt.zero_grad()
        return {"prediction": y_hat}
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x[:,:,0:1].squeeze(2).float()
        y = y[:,:,0:1].squeeze(2).float()

        x[torch.isnan(x)] = self.loss_ignore_value
        y[torch.isnan(y)] = self.loss_ignore_value

        y_hat, mf = self(x)

        loss = 0
        phys_loss = 0
        cmax_loss = 0

        sequence = torch.cat([x, y], dim=1).squeeze(1)
        for i in range(sequence.shape[1]-1):
            extrapolated = self._extrapolate_3d(1, sequence[:,i:i+1], mf, self.trainer.state.stage)
            loss += self.criterion(extrapolated, sequence[:,i+1:i+2])
            cmax_loss += self.criterion(torch.max(extrapolated, 2).values, torch.max(sequence[:,i+1:i+2], 2).values)

        phys_loss = self.phys_loss(mf)
        loss = loss / (sequence.shape[1] - 1)
        cmax_loss = cmax_loss / (sequence.shape[1] - 1)

        if self.ground_loss_weight > 0:
            neg_mask = mf[:,0,0] < 0
            ground_loss = torch.sum(mf[:,0,0][neg_mask] ** 2) / (mf.shape[0] * mf.shape[-2] * mf.shape[-1])
            ground_loss = ground_loss * self.ground_loss_weight
            loss = loss + ground_loss
            self.log("val_ground_loss", ground_loss.detach())

        cmax_extra_loss = self.criterion(torch.max(y_hat, 2).values, torch.max(y, 2).values)

        if not self.trainer.sanity_checking:
            self.log("val_loss", loss.detach() + (phys_loss.detach() * self.beta) + (cmax_loss.detach() * self.cmax_loss_weight))
            self.log("val_crit_loss", loss.detach())
            self.log("val_phys_loss", phys_loss.detach())
            self.log("val_cmax_loss", cmax_loss.detach())
            self.log("val_extra_loss", cmax_extra_loss.detach())

        if batch_idx == 0:
            print("Logging nowcast images")
            y_hat = self.trainer.datamodule.valid_dataset.invScaler(y_hat)
            y_hat[y == self.loss_ignore_value] = torch.nan
            fig, axes = plt.subplots(4, 4, figsize=(16,16), layout='constrained')
            for i in range(4):
                for j in range(4):
                    plot_precip_field(y_hat[:4,0:16:4,0][i][j].cpu().numpy(), colorbar=False, units='mm/h', ax=axes[i,j])
                    quiver(mf.cpu().numpy()[:,2:0:-1,0][i], step=16, ax=axes[i,j])
                    axes[i,j].set_xticks(np.arange(0, y_hat.shape[-2], 64))
                    axes[i,j].set_yticks(np.arange(0, y_hat.shape[-1], 64))
                    axes[i,j].grid(True)

            self.logger.log_image(key="nowcasts_valid_pysteps", images=[fig])

        return {"prediction": y_hat, "val_loss": loss}
    
    def on_validation_epoch_start(self):
        if self.trainer.ckpt_path is not None and self.trainer.fit_loop.restarting:
            self.trainer.sanity_checking = True

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        elif isinstance(sch, torch.optim.lr_scheduler.ExponentialLR):
            sch.step()
    
    @staticmethod
    def _get_conf_mat(pred, target, threshold):
        pred_mask = pred.flatten() > threshold
        target_mask = target.flatten() > threshold
        tn, fp, fn, tp = torch.bincount(target_mask*2 + pred_mask, minlength=4)
        return tn, fp, fn, tp

    @staticmethod
    def _get_disc_metrics(tn, fp, fn, tp):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        # accuracy = (tp + tn) / (tp + tn + fp + fn)
        a_r = ((tp + fp) * (tp + fn))/(tp + fp + tn + fn)
        ets = (tp - a_r)/(tp + fp + fn - a_r)
        # far = fp / (tp + fp)
        return precision, recall, ets

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x[:,:,0:1].squeeze(2).float()
        y = y[:,:,0:1].squeeze(2).float()

        x[torch.isnan(x)] = self.loss_ignore_value
        
        y_hat, mf = self(x)
        y_hat = y_hat[:,:self.predict_leadtimes]

        y_hat = self.trainer.datamodule.test_dataset.invScaler(y_hat.unsqueeze(1)).squeeze()
        y_hat[y_hat < 0] = 0
        y_hat[torch.isnan(y)] = 0

        y = y[:,:self.predict_leadtimes]
        y = self.trainer.datamodule.test_dataset.invScaler(y.unsqueeze(1)).squeeze()
        y[torch.isnan(y)] = 0

        for i in range(self.predict_leadtimes):
            CMAX_center_pred = TF.center_crop(torch.max(y_hat[:,i], 1).values, 320)
            CMAX_center_target = TF.center_crop(torch.max(y[:,i], 1).values, 320)

            mse = torch.nn.functional.mse_loss(CMAX_center_pred, CMAX_center_target).detach()
            self.log(f"test/CMAX_MSE-{i:02d}", mse)

            mae = torch.nn.functional.l1_loss(CMAX_center_pred, CMAX_center_target).detach()
            self.log(f"test/CMAX_MAE-{i:02d}", mae)

            me = (CMAX_center_pred - CMAX_center_target).mean().detach()
            self.log(f"test/CMAX_ME-{i:02d}", me)

            for threshold in [1, 5, 10]:
                tn, fp, fn, tp = self._get_conf_mat(CMAX_center_pred, CMAX_center_target, threshold)
                self.log(f"test/CMAX_TN-{threshold}mmh-{i:02d}", tn.float().detach(), reduce_fx=torch.sum)
                self.log(f"test/CMAX_FP-{threshold}mmh-{i:02d}", fp.float().detach(), reduce_fx=torch.sum)
                self.log(f"test/CMAX_FN-{threshold}mmh-{i:02d}", fn.float().detach(), reduce_fx=torch.sum)
                self.log(f"test/CMAX_TP-{threshold}mmh-{i:02d}", tp.float().detach(), reduce_fx=torch.sum)
        
        if batch_idx % 100 == 0:
            print("Logging nowcast images")
            fig, axes = plt.subplots(4, 4, figsize=(16,16), layout='constrained')
            for i in range(4):
                for j in range(4):
                    plot_precip_field(y_hat[:4,0:16:4,0][i][j].cpu().numpy(), colorbar=False, units='mm/h', ax=axes[i,j])
                    quiver(mf.cpu().numpy()[:,2:0:-1,0][i], step=16, ax=axes[i,j])
                    axes[i,j].set_xticks(np.arange(0, y_hat.shape[-2], 64))
                    axes[i,j].set_yticks(np.arange(0, y_hat.shape[-1], 64))
                    axes[i,j].grid(True)

            self.logger.log_image(key="nowcasts_test_pysteps", images=[fig])

        return {"prediction": y_hat}
    
    def on_test_epoch_end(self):
        leadtimes = list(range(5, (self.predict_leadtimes + 1) * 5, 5))
        test_metrics_names = ['CMAX_MSE', 'CMAX_MAE', 'CMAX_ME']
        test_metrics_thresholded_names = ['CMAX_Precision', 'CMAX_Recall', 'CMAX_ETS']
        test_metrics = {}
        for name in test_metrics_names:
            test_metrics[name] = []
        for name in test_metrics_thresholded_names:
            for threshold in [1, 5, 10]:
                test_metrics[f"{name}-{threshold}mmh"] = []

        for i in range(self.predict_leadtimes):
            for name in test_metrics_names:
                test_metrics[name].append(self.trainer.callback_metrics.get(f"test/{name}-{i:02d}").cpu().item())
        
        for i in range(self.predict_leadtimes):
            for threshold in [1, 5, 10]:
                tn = self.trainer.callback_metrics.get(f"test/CMAX_TN-{threshold}mmh-{i:02d}").cpu().item()
                fp = self.trainer.callback_metrics.get(f"test/CMAX_FP-{threshold}mmh-{i:02d}").cpu().item()
                fn = self.trainer.callback_metrics.get(f"test/CMAX_FN-{threshold}mmh-{i:02d}").cpu().item()
                tp = self.trainer.callback_metrics.get(f"test/CMAX_TP-{threshold}mmh-{i:02d}").cpu().item()

                precision, recall, ets = self._get_disc_metrics(tn, fp, fn, tp)

                test_metrics[f"CMAX_Precision-{threshold}mmh"].append(precision)
                test_metrics[f"CMAX_Recall-{threshold}mmh"].append(recall)
                test_metrics[f"CMAX_ETS-{threshold}mmh"].append(ets)
        
        self.logger.log_table(key="test_metrics",
                              columns=['leadtime', *test_metrics.keys()],
                              data=list(map(list, zip(*[leadtimes, *[test_metrics[name] for name in test_metrics.keys()]]))))
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, _ = batch
        x = x[:,:,0:1].permute(0,2,1,3,4).float()
        y = y[:,:,0:1].permute(0,2,1,3,4).float()

        out_mask = torch.isnan(y).squeeze(1)

        x[torch.isnan(x)] = self.loss_ignore_value
        y[torch.isnan(y)] = self.loss_ignore_value

        y_hat, mf = self(x)
        y_hat = y_hat[:,:self.predict_leadtimes]
        y_hat[y_hat < 0] = 0

        y_hat = self.trainer.datamodule.predict_dataset.from_transformed(
            y_hat.unsqueeze_(1)
        )

        y_hat = y_hat.squeeze(1)
        y_hat[out_mask] = torch.nan

        del x
        return y_hat
    
    @staticmethod
    def _extrapolate_3d(timesteps, precip, motion_field, stage):
        """
        Extrapolates 3D precipitation data using a 3D motion field.
        Args:
            timesteps (int): Number of timesteps to extrapolate.
            precip (torch.Tensor): Precipitation tensor (B, C, D, H, W)
            motion_field (torch.Tensor): Motion field tensor (B, 3, D, H, W) for (dz, dy, dx)
            stage (str): The stage of the model ("train", "predict", "test", etc.)
        Returns:
            torch.Tensor: Extrapolated precipitation data (B, timesteps, D, H, W)
        """
        if not isinstance(precip, torch.Tensor) or precip.dim() != 5:
            raise ValueError("precip must be a 5D tensor (B, C, D, H, W)")
        if not isinstance(motion_field, torch.Tensor) or motion_field.dim() != 5 or motion_field.shape[1] != 3:
            raise ValueError("motion_field must be a 5D tensor (B, 3, D, H, W)")
        if timesteps <= 0:
            raise ValueError("timesteps must be a positive integer")
        if precip.shape[0] != motion_field.shape[0]:
            raise ValueError("Batch sizes must match")
        if precip.shape[2:] != motion_field.shape[2:]:
            raise ValueError("Spatial dimensions must match")

        B, C, X, Y, Z = precip.shape
        device = precip.device

        # Normalize motion field for grid_sample
        velocity = motion_field
        velocity = torch.stack([
            velocity[:, 2] / ((Z-1) / 2),
            velocity[:, 1] / ((Y-1) / 2),
            velocity[:, 0] / ((X-1) / 2)
        ], dim=1)

        x_values, y_values, z_values = torch.meshgrid(torch.arange(X), torch.arange(Y), torch.arange(Z))
        xyz_coords = torch.stack([z_values, y_values, x_values]).to(precip.device).float()
        xyz_coords[0] = ((xyz_coords[0]) / ((Z - 1) / 2) - 1)
        xyz_coords[1] = ((xyz_coords[1]) / ((Y - 1) / 2) - 1)
        xyz_coords[2] = ((xyz_coords[2]) / ((X - 1) / 2) - 1)

        precip_extrap = torch.zeros((B, timesteps, X, Y, Z), device=device)
        displacement = torch.zeros_like(velocity, device=device)
        velocity_inc = velocity.clone()

        for ti in range(timesteps):
            coords_warped = xyz_coords + displacement  # (B, 3, D, H, W)
            
            velocity_inc = F.grid_sample(
                velocity, coords_warped.movedim(1, -1), mode='bilinear',
                padding_mode='border', align_corners=True
            )
            displacement -= velocity_inc
            
            coords_warped = xyz_coords + displacement

            sample_mode = 'nearest' if stage in ["predict", "test", "validate"] else 'bilinear'
            precip_warped = F.grid_sample(
                precip, coords_warped.movedim(1, -1), mode=sample_mode,
                padding_mode='border', align_corners=True
            )
            precip_extrap[:, ti] = precip_warped.squeeze(1)

        return precip_extrap


class PhysLoss(nn.Module):
    """Physics-informed loss function for the motion field."""
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[[[-1.,  0.,  1.],
                                       [-2.,  0.,  2.],
                                       [-1.,  0.,  1.]],
                                      [[-2.,  0.,  2.],
                                       [-4.,  0.,  4.],
                                       [-2.,  0.,  2.]],
                                      [[-1.,  0.,  1.],
                                       [-2.,  0.,  2.],
                                       [-1.,  0.,  1.]]]]).view(1, 1, 3, 3, 3).repeat(1, 1, 1, 1, 1)

        self.sobel_y = torch.tensor([[[[-1., -2., -1.],
                                       [ 0.,  0.,  0.],
                                       [ 1.,  2.,  1.]],
                                      [[-2., -4., -2.],
                                       [ 0.,  0.,  0.],
                                       [ 2.,  4.,  2.]],
                                      [[-1., -2., -1.],
                                       [ 0.,  0.,  0.],
                                       [ 1.,  2.,  1.]]]]).view(1, 1, 3, 3, 3).repeat(1, 1, 1, 1, 1)

        self.sobel_z = torch.tensor([[[[-1., -2., -1.],
                                       [-2., -4., -2.],
                                       [-1., -2., -1.]],
                                      [[ 0.,  0.,  0.],
                                       [ 0.,  0.,  0.],
                                       [ 0.,  0.,  0.]],
                                      [[ 1.,  2.,  1.],
                                       [ 2.,  4.,  2.],
                                       [ 1.,  2.,  1.]]]]).view(1, 1, 3, 3, 3).repeat(1, 1, 1, 1, 1)
        
    def forward(self, motion_field):
        """
        Computes the physics-informed conservation of mass loss for the motion field.
        Args:
            motion_field (torch.Tensor): The motion field tensor of shape (batch_size, 3, height, width, depth).
        Returns:
            torch.Tensor: The physics-informed conservation of mass loss.
        """
        if not isinstance(motion_field, torch.Tensor):
            raise TypeError("motion_field must be a torch.Tensor")
        if motion_field.dim() != 5 or motion_field.shape[1] != 3:
            raise ValueError("motion_field must be a 5D tensor with shape (batch_size, 3, height, width)")
        device = motion_field.device
        
        # physics-informed conservation of mass loss
        motion_field = F.pad(motion_field, (1, 1, 1, 1, 1, 1), mode='replicate')
        diff_u = F.conv3d(motion_field[:,0:1], self.sobel_x.to(device))
        diff_v = F.conv3d(motion_field[:,1:2], self.sobel_y.to(device))
        diff_w = F.conv3d(motion_field[:,2:3], self.sobel_z.to(device))
        physics_loss = torch.sum(torch.abs(diff_u + diff_v + diff_w)) / (motion_field.shape[0] * motion_field.shape[-3] * motion_field.shape[-2] * motion_field.shape[-1])
        
        return physics_loss


class CustomMSELoss(nn.Module):
    """Custom MSE loss function that ignores -1 values in the target tensor."""
    def __init__(self, ignore_value=-1):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.ignore_value = ignore_value
    
    def forward(self, pred, target):
        """
        Computes the MSE loss while ignoring -1 values in the target tensor.
        Args:
            pred (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The target tensor, where -1 values are ignored.
        Returns:
            torch.Tensor: The mean squared error loss, ignoring -1 values in the target.
        """
        datamask = (target != self.ignore_value)
        loss = self.mse(pred, target) * datamask
        return torch.mean(loss)
