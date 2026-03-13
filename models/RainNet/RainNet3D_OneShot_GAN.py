"""RainNet iterative model definition with definitions of custom loss functions."""

import numpy as np
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

from modelcomponents import RainNet3D as RN
from modelcomponents.NowcastNet.TemporalDiscriminator import TemporalDiscriminator as td

import matplotlib.pyplot as plt
from pysteps.visualization import plot_precip_field
from pysteps.verification import fss

class RainNet3D_OneShot_GAN(pl.LightningModule):
    """Model for the RainNet iterative neural network."""

    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()

        self.config = config

        self.personal_device = torch.device(config.train_params.device)
        self.gen_network = RN(
            kernel_size=config.model.rainnet.kernel_size,
            mode=config.model.rainnet.mode,
            im_shape=config.model.rainnet.output_shape,
            conv_shape=config.model.rainnet.conv_shape,
        )

        self.disc_network = td()

        if config.model.loss.name == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss {config.model.loss.name} not implemented!")

        self.d_loss = nn.BCEWithLogitsLoss()

        self.crit_disc_coeff = config.model.loss.crit_disc_coeff
        self.lambda_w = config.model.loss.lambda_w
        

        # on which leadtime to train the NN on?
        self.train_leadtimes = config.model.train_leadtimes
        self.verif_leadtimes = config.train_params.verif_leadtimes
        # How many leadtimes to predict
        self.predict_leadtimes = config.prediction.predict_leadtimes

        # optimization parameters
        self.lr = float(config.model.lr)
        self.disc_lr_coef = config.model.disc_lr_coef
        self.lr_sch_params = config.train_params.lr_scheduler
        self.automatic_optimization = False


    def forward(self, x):
        return self.gen_network(x)

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.gen_network.parameters(), lr=self.lr)
        optimizer_D = torch.optim.Adam(self.disc_network.parameters(), lr=self.lr * self.disc_lr_coef)
        if self.lr_sch_params.name is None:
            return [optimizer_G, optimizer_D]
        elif self.lr_sch_params.name == "reduce_lr_on_plateau":
            lr_scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer_G, **self.lr_sch_params.kwargs
            )
            lr_scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer_D, **self.lr_sch_params.kwargs
            )
            return [optimizer_G, optimizer_D], [lr_scheduler_G, lr_scheduler_D]
        elif self.lr_sch_params.name == "ExponentialLR":
            lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer_G, **self.lr_sch_params.kwargs
            )
            lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer_D, **self.lr_sch_params.kwargs
            )
            return [optimizer_G, optimizer_D], [lr_scheduler_G, lr_scheduler_D]
        else:
            raise NotImplementedError("Lr scheduler not defined.")


    
    def on_fit_start(self):
        if self.trainer.ckpt_path:
            self.trainer.early_stopping_callback.patience = self.config.train_params.early_stopping.patience


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        x, y, _ = batch
        x = x[:,:,0:1].permute(0,2,1,3,4).float()
        y = y[:,:,0:1].permute(0,2,1,3,4).float()

        x[torch.isnan(x)] = 0
        y[torch.isnan(y)] = 0

        x[x < 0.08] = 0
        y[y < 0.08] = 0
    
        optimizer_g, optimizer_d = self.optimizers()

        # train generative postprocess
        # generate base LUPIN images
        self.toggle_optimizer(optimizer_g)
        y_hat = self(x)

        x = x.squeeze(1)
        y = y.squeeze(1)
        y_hat = y_hat.squeeze(2)

        s_loss = self.criterion(y_hat, y)
        
        # adversarial loss is binary cross-entropy
        g_loss = 0
        D_G1 = self.disc_network(torch.cat([x, y_hat], dim=1)).view(-1)
        g_loss += self.d_loss(D_G1, torch.ones(D_G1.shape, device=D_G1.device))

        self.manual_backward(g_loss * (1 - self.crit_disc_coeff) + s_loss * self.crit_disc_coeff)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)
        y_hat = y_hat.detach()

        D_R = self.disc_network(torch.cat([x, y], dim=1)).view(-1)
        #real_loss = self.d_loss(D_R, torch.ones(D_R.shape, device=D_R.device))

        D_G2 = self.disc_network(torch.cat([x, y_hat], dim=1)).view(-1)
        #fake_loss = self.d_loss(D_G2, torch.zeros(D_G2.shape, device=D_G2.device))

        d_loss = D_G2.mean() - D_R.mean()

        if self.lambda_w > 0:

            # Gradient penalty
            alpha = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            alpha = alpha.expand_as(y)
            interpolated = alpha * y + (1 - alpha) * y_hat
            interpolated = interpolated.requires_grad_(True)
            D_interp = self.disc_network(torch.cat([x, interpolated], dim=1)).view(-1)

            gradients = torch.autograd.grad(
                outputs=D_interp,
                inputs=interpolated,
                grad_outputs=torch.ones_like(D_interp),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            d_loss += self.lambda_w * gradient_penalty

            self.log("Gradient_Penalty", gradient_penalty)

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        self.log_dict({"Disc_GenLabel_Mean": D_G1.mean(), "Gen_C-E_Loss": g_loss, "Disc_RealLabel_Mean": D_R.mean(), "Disc_C-E_Loss": d_loss, "Gen_Crit_Loss": s_loss})

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y, _ = batch
            x = x[:,:,0:1].permute(0,2,1,3,4).float()
            y = y[:,:,0:1].permute(0,2,1,3,4).float()

            x[torch.isnan(x)] = 0
            y[torch.isnan(y)] = 0

            x[x < 0.08] = 0
            y[y < 0.08] = 0
        
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
        sch_G, sch_D = self.lr_schedulers()
        if isinstance(sch_G, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch_G.step(self.trainer.callback_metrics["val_loss"])
            sch_D.step(self.trainer.callback_metrics["val_loss"])
        elif isinstance(sch_G, torch.optim.lr_scheduler.ExponentialLR):
            sch_G.step()
            sch_D.step()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y, _ = batch
            x = x[:,:,0:1].permute(0,2,1,3,4).float()
            y = y[:,:,0:1].permute(0,2,1,3,4).float()

            x[torch.isnan(x)] = 0
            y[torch.isnan(y)] = 0

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
            x = x[:,:,0:1].permute(0,2,1,3,4).float()
            y = y[:,:,0:1].permute(0,2,1,3,4).float()

            out_mask = torch.isnan(y).squeeze(1)

            x[torch.isnan(x)] = 0
            y[torch.isnan(y)] = 0

            y_hat = self(x)[:,:self.predict_leadtimes]
            y_hat[y_hat < 0] = 0

        y_hat = self.trainer.datamodule.predict_dataset.from_transformed(
            y_hat.permute(0,2,1,3,4)
        ).permute(0,2,1,3,4)

        y_hat = y_hat.squeeze(2)
        y_hat[out_mask] = torch.nan

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
    