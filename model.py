from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
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
            "loss": loss + self.lambda_arb * (delta_loss + theta_loss + gamma_loss + dual_delta_loss + dual_gamma_loss),
            "mse_loss": loss,
            "delta_loss": delta_loss,
            "gamma_loss": gamma_loss,
            "theta_loss": theta_loss,
            "dual_delta_loss": dual_delta_loss,
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
            #layers.append(nn.Tanh())
            input_dims = n_hidden

        layers.append(nn.Linear(n_hidden, 1))
        layers.append(nn.Softplus())

        self.model = nn.Sequential(*layers)

    def compute_greeks(self, x: torch.Tensor):

        s = x[:, 0:1].requires_grad_(True)
        k = x[:, 1:2].requires_grad_(True)
        t = x[:, 2:3].requires_grad_(True)
        rest = x[:, 3:].requires_grad_(True)

        m = s / k

        model_input = torch.cat([m, t, rest], dim=1)

        normalized_prices = self.model(model_input)

        prices = normalized_prices * k

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

        return {
            "delta": delta,
            "gamma": gamma,
            "dual_delta": dual_delta,
            "dual_gamma": dual_gamma,
            "theta": theta,
            "vega": 0,
            "rho": 0
        }

    def forward(self, x):

        s = x[:, 0:1]
        k = x[:, 1:2]
        t = x[:, 2:3]
        rest = x[:, 3:]

        m = s / k

        model_input = torch.cat([m, t, rest], dim=1)

        predictions = self.model(model_input)

        greeks = self.compute_greeks(x)

        return predictions, greeks

class OptionNetModule(pl.LightningModule):
    def __init__(self, n_inputs=3, n_hidden=64, n_layers=4, lambda_arb = 1.0, learning_rate=1e-3, example_inputs = None):
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

        self.lambda_arb = lambda_arb
        self.example_input_array = example_inputs

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
        self.log("train_dual_delta_loss", loss["dual_delta_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("train_greeks_loss", self.lambda_arb * (loss["mse_loss"] +
                loss["delta_loss"]+
                loss["gamma_loss"]+
                loss["theta_loss"]+
                loss["dual_gamma_loss"] + loss["dual_delta_loss"]), on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def on_validation_model_eval(self) -> None:
        super().on_validation_model_eval()
        torch.set_grad_enabled(True)

    def on_test_model_eval(self) -> None:
        super().on_test_model_eval()
        torch.set_grad_enabled(True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)

        loss = self.loss_fn(y_pred, y)

        self.log("test_loss", loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_mse_loss", loss["mse_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_delta_loss", loss["delta_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_gamma_loss", loss["gamma_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_theta_loss", loss["theta_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_dual_gamma_loss", loss["dual_gamma_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("test_dual_delta_loss", loss["dual_delta_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("test_greeks_loss", self.lambda_arb * (loss["mse_loss"] +
                                                       loss["delta_loss"] +
                                                       loss["gamma_loss"] +
                                                       loss["theta_loss"] +
                                                       loss["dual_gamma_loss"] + loss["dual_delta_loss"]),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)

        loss = self.loss_fn(y_pred, y)

        self.log("val_loss", loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mse_loss", loss["mse_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_delta_loss", loss["delta_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_gamma_loss", loss["gamma_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_theta_loss", loss["theta_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_dual_gamma_loss", loss["dual_gamma_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("val_dual_delta_loss", loss["dual_delta_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("val_greeks_loss", self.lambda_arb * (loss["mse_loss"] +
                                                         loss["delta_loss"] +
                                                         loss["gamma_loss"] +
                                                         loss["theta_loss"] +
                                                         loss["dual_gamma_loss"] + loss["dual_delta_loss"]),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [StepLR(optimizer, step_size=7, gamma=0.9), scheduler]





