from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import lightning as pl
from torch.optim import Adam


class GreeksInformedLoss(nn.Module):
    def __init__(self, lambda_arb = 1.0):
        super().__init__()
        self.loss = nn.MSELoss()
        self.lambda_arb = lambda_arb

    def forward(self, y_pred, y_true):
        pred, greeks = y_pred

        loss = self.loss(pred, y_true)

        delta = greeks["delta"]
        dual_delta = greeks["dual_delta"]
        gamma = greeks["gamma"]
        theta = greeks["theta"]
        dual_gamma = greeks["dual_gamma"]


        delta_loss = torch.mean(torch.relu(-delta) ** 2)
        dual_delta_loss = torch.mean(torch.relu(dual_delta) ** 2)
        gamma_loss = torch.mean(torch.relu(-gamma) ** 2)
        theta_loss = torch.mean(torch.relu(-theta) ** 2)
        dual_gamma_loss = torch.mean(torch.relu(-dual_gamma) ** 2)

        return {
            "loss": loss + self.lambda_arb * (delta_loss + theta_loss + gamma_loss + dual_delta_loss + dual_gamma_loss),#(delta_loss + gamma_loss + theta_loss + dual_delta_loss + dual_gamma_loss),
            "mse_loss": loss,
            "delta_loss": delta_loss,
            "gamma_loss": gamma_loss,
            "theta_loss": theta_loss,
            "dual_gamma_loss": dual_gamma_loss,
        }

class OptionNet(nn.Module):
    def __init__(self, n_inputs=3, n_hidden=64, n_layers=4):
        super().__init__()

        layers = []
        input_dims = n_inputs

        for _ in range(n_layers):
            layers.append(
                nn.Linear(input_dims, n_hidden)
            )
            layers.append(nn.Softplus())
            input_dims = n_hidden

        layers.append(nn.Linear(n_hidden, 1))
        #layers.append(nn.Softplus())

        self.model = nn.Sequential(*layers)

    def compute_greeks(self, x: torch.Tensor):

        s = x[:, 0:1].requires_grad_(True)
        k = x[:, 1:2].requires_grad_(True)
        t = x[:, 2:3].requires_grad_(True)
        rest = x[:, 3:].requires_grad_(True)

        m = s / k

        model_input = torch.cat([m, t, rest], dim=1)

        prices = self.model(model_input)

        grads = torch.autograd.grad(
            outputs=prices,
            inputs=[s, k, t],
            grad_outputs=torch.ones_like(prices),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )

        delta = grads[0]
        dual_delta = grads[1]
        theta = grads[2]

        gamma = torch.autograd.grad(
            outputs=delta,
            inputs=s,
            grad_outputs=torch.ones_like(delta),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        dual_gamma = torch.autograd.grad(
            outputs=dual_delta,
            inputs=k,
            grad_outputs=torch.ones_like(dual_delta),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Input Index Map: 0=S, 1=K, 2=T, 3=sigma, 4=r
        return {
            "delta": delta, # d/dS
            "gamma": gamma, # d^2/dS^2 (Diagonal of Hessian at 0,0)
            "dual_delta": dual_delta,
            "dual_gamma": dual_gamma,
            "theta": theta, # d/dT
            "vega": 0,#pred_grads[:, 3:4], # d/dSigma
            "rho": 0#pred_grads[:, 4:5], # d/dR
        }

    def forward(self, x):

        s = x[:, 0:1]
        k = x[:, 1:2]
        t = x[:, 2:3]
        rest = x[:, 3:]

        m = s / k

        model_input = torch.cat([m, t, rest], dim=1)

        predictions = self.model(model_input)
        if self.training:
            greeks = self.compute_greeks(x)
        else:
            greeks = {
                "delta": torch.tensor(0, dtype=torch.float32), # d/dS
                "gamma": torch.tensor(0, dtype=torch.float32) * 1e1, # d^2/dS^2 (Diagonal of Hessian at 0,0)
                "dual_gamma": torch.tensor(0, dtype=torch.float32),  # d^2/dK^2 (Diagonal of Hessian at 1,1)
                "theta": torch.tensor(0, dtype=torch.float32), # d/dT
                "vega": 0,#pred_grads[:, 3:4], # d/dSigma
                "rho": 0#pred_grads[:, 4:5], # d/dR
            }

        return predictions, greeks

class OptionNetModule(pl.LightningModule):
    def __init__(self, n_inputs=3, n_hidden=64, n_layers=4, lambda_arb = 1.0, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = OptionNet(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_layers=n_layers,
        )

        self.loss_fn = GreeksInformedLoss(
            lambda_arb=lambda_arb
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)

        loss = self.loss_fn(y_pred, y)

        self.log("train_loss", loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mse_loss", loss["mse_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_delta_loss", loss["delta_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_gamma_loss", loss["gamma_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_theta_loss", loss["theta_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dual_gamma_loss", loss["dual_gamma_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_greeks_loss", loss["mse_loss"] +
                loss["delta_loss"]+
                loss["gamma_loss"]+
                loss["theta_loss"]+
                loss["dual_gamma_loss"] + loss["dual_gamma_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)

        #loss = self.loss_fn(y_pred, y)
        loss = self.loss_fn.loss(y_pred[0], y)

        self.log("test_loss", loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_mse_loss", loss["mse_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("test_delta_loss", loss["delta_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("test_gamma_loss", loss["gamma_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("test_theta_loss", loss["theta_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("test_dual_gamma_loss", loss["dual_gamma_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)

        #loss = self.loss_fn(y_pred, y)
        loss = self.loss_fn.loss(y_pred[0], y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mse_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_delta_loss", loss["delta_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_gamma_loss", loss["gamma_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_theta_loss", loss["theta_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_dual_gamma_loss", loss["dual_gamma_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)

        # scheduler = {
        #     'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
        #     'monitor': 'val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }

        return [optimizer], [StepLR(optimizer, step_size=5, gamma=0.9)]





