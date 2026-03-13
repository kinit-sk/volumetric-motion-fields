"""RainNet iterative model definition with definitions of custom loss functions."""

import numpy as np
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

from modelcomponents import RainNet3D as RN

import matplotlib.pyplot as plt
from pysteps.visualization import plot_precip_field
from pysteps.verification import fss

class RainNet3D_OneShot(pl.LightningModule):
    """Model for the RainNet iterative neural network."""

    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()

        self.personal_device = torch.device(config.train_params.device)
        self.network = RN(
            kernel_size=config.model.rainnet.kernel_size,
            mode=config.model.rainnet.mode,
            im_shape=config.model.rainnet.output_shape,
            conv_shape=config.model.rainnet.conv_shape,
        )

        if config.model.loss.name == "mse":
            self.criterion = CustomMSELoss()
        else:
            raise NotImplementedError(f"Loss {config.model.loss.name} not implemented!")
        self.loss_precip_only = config.model.loss.precip_only
        

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
        return self.network(x)

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
    
    def on_train_epoch_start(self):
        pass


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        x, y, _ = batch
        x = x[:,:,0:1].permute(0,2,1,3,4).float()
        y = y[:,:,0:1].permute(0,2,1,3,4).float()

        x[torch.isnan(x)] = -1
        y[torch.isnan(y)] = -1

        y_hat = self(x)
        
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
            
        self.manual_backward(loss)

        opt.step()
        opt.zero_grad()
        self.log("train_loss", loss.detach())
        return {"prediction": y_hat, "loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y, _ = batch
            x = x[:,:,0:1].permute(0,2,1,3,4).float()
            y = y[:,:,0:1].permute(0,2,1,3,4).float()

            x[torch.isnan(x)] = -1
            y[torch.isnan(y)] = -1

            y_hat = self(x)[:,:self.verif_leadtimes]
            
            loss = self.criterion(y_hat.squeeze(), y.squeeze()) #* loss_weights[i]
            loss = loss.detach()
        
        if not self.trainer.sanity_checking:
            self.log("val_loss", loss)

        if batch_idx == 0:
            print("Logging nowcast images")
            grid = torchvision.utils.make_grid(y_hat[:4,0:16:4,0].flatten(0,1).unsqueeze(1), nrow=4, padding=8, normalize=True, value_range=(0,10))
            self.logger.log_image(key="nowcasts_valid_torchvision", images=[grid])

            fig, axes = plt.subplots(4, 4, figsize=(16,16), layout='constrained')
            for i in range(4):
                for j in range(4):
                    plot_precip_field(y_hat[:4,0:16:4,0][i][j].cpu().numpy(), colorbar=False, units='mm/h', ax=axes[i,j])

            self.logger.log_image(key="nowcasts_valid_pysteps", images=[fig])

        return {"prediction": y_hat, "loss": loss}
    
    def on_validation_epoch_start(self):
        if self.trainer.ckpt_path is not None and self.trainer.fit_loop.restarting:
            self.trainer.sanity_checking = True

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        sch = self.lr_schedulers()
        if not self.trainer.sanity_checking:
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(self.trainer.callback_metrics["val_loss"])
            elif isinstance(sch, torch.optim.lr_scheduler.ExponentialLR):
                sch.step()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y, _ = batch
            x = x[:,:,0:1].permute(0,2,1,3,4).float()
            y = y[:,:,0:1].permute(0,2,1,3,4).float()

            x[torch.isnan(x)] = -1
            y[torch.isnan(y)] = -1

            y_hat = self(x)[:,:self.verif_leadtimes]
            y_hat[y_hat < 0] = 0
            
            for i in range(self.predict_leadtimes):
                custom_mse = CustomMSELoss()
                mse = custom_mse(y_hat[:, i], y[:, :, i]).detach()
                self.log(f"test/mse{i}", mse)

                if (i == 5) or (i == 11):
                    for t in [1,5,10]:
                        for s in [1,4,16]:
                            forecast = y_hat[:, i].squeeze().cpu().numpy()
                            observation = y[:, :, i].squeeze().cpu().numpy()
                            for j in range(len(forecast)):
                                self.log(f"test/fss_thr{t}_scale{s}_lead{(i+1)*5}", fss(forecast[j], observation[j], thr=t, scale=s))

        return {"prediction": y_hat}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            x, y, _ = batch
            x = x.float()
            y = y.float()

            y_hat = self(x)[:,:self.predict_leadtimes]

        y_hat = self.trainer.datamodule.predict_dataset.from_transformed(
            y_hat.permute(0,1,3,4,2)
        ).permute(0,1,4,2,3)

        del x
        return y_hat
    
class CustomMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    
    def forward(self, pred, target):
        datamask = (target != -1)
        loss = self.mse(pred, target) * datamask
        return torch.mean(loss)
    