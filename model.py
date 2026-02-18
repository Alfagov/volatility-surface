from typing import Any, Tuple, Dict

import torch
import torch.nn as nn
import torchmetrics.functional
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import lightning as pl
from torch.optim import Adam
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredLogError


class GreeksInformedLoss(nn.Module):
    def __init__(
            self,
            lambda_arb = 1.0,
            theta_floor_base: float = -0.03,
            theta_floor_slope: float = 0.0393,
            theta_floor_eps: float = 1e-4,
            delta_ceiling: float = 0.9999,
            price_spot_margin: float = 1e-4,
            w_delta: float = 1.0,
            w_delta_upper: float = 1.0,
            w_dual_delta: float = 1.0,
            w_gamma: float = 1.0,
            w_theta: float = 1.0,
            w_theta_upper: float = 1.0,
            w_dual_gamma: float = 1.0,
            w_price_upper: float = 1.0,
            vega_weight_eps: float = 1e-8,
    ):
        super().__init__()
        self.price_loss = nn.MSELoss()
        self.lambda_arb = lambda_arb

        self.theta_floor_base = theta_floor_base
        self.theta_floor_slope = theta_floor_slope
        self.theta_floor_eps = theta_floor_eps
        self.delta_ceiling = delta_ceiling
        self.price_spot_margin = price_spot_margin

        self.w_delta = w_delta
        self.w_delta_upper = w_delta_upper
        self.w_dual_delta = w_dual_delta
        self.w_gamma = w_gamma
        self.w_theta = w_theta
        self.w_theta_upper = w_theta_upper
        self.w_dual_gamma = w_dual_gamma
        self.w_price_upper = w_price_upper
        self.vega_weight_eps = vega_weight_eps

        self.bucket_weights = {
            "deep_otm": 1.0,
            "otm": 1.0,
            "atm": 1.5,
            "itm": 3.0,
            "deep_itm": 4.0,
        }

    def _moneyness_weights(self, s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        m = s / torch.clamp(k, min=1e-8)
        w = torch.ones_like(m)

        w = torch.where(m < 0.90, torch.full_like(w, self.bucket_weights["deep_otm"]), w)
        w = torch.where((m >= 0.90) & (m < 0.97), torch.full_like(w, self.bucket_weights["otm"]), w)
        w = torch.where((m >= 0.97) & (m < 1.03), torch.full_like(w, self.bucket_weights["atm"]), w)
        w = torch.where((m >= 1.03) & (m < 1.10), torch.full_like(w, self.bucket_weights["itm"]), w)
        w = torch.where(m >= 1.10, torch.full_like(w, self.bucket_weights["deep_itm"]), w)
        return w

    def weighted_mse_loss(self, out, target):
        # Add small epsilon to prevent div by zero
        epsilon = 1e-6
        # Calculate the squared error
        squared_error = (out - target) ** 2
        # Weight by the squared target value (effectively calculating % error squared)
        weights = 1 / (target ** 2 + epsilon)

        return torch.mean(weights * squared_error)

    def forward(
            self,
            y_pred: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
            y_true: torch.Tensor,
            t: torch.Tensor,
            s: torch.Tensor,
            k: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        pred, greeks = y_pred
        # mse_loss = self.price_loss(pred, y_true)
        #
        # w = self._moneyness_weights(s, k)
        # price_loss_vec = mse_loss * w
        #
        # mse_loss = price_loss_vec.mean()

        delta = greeks["delta"]
        dual_delta = greeks["dual_delta"]
        gamma = greeks["gamma"]
        theta = greeks["theta"]
        dual_gamma = greeks["dual_gamma"]
        vega = greeks["vega"]
        call_prices = pred * k
        #squared_error = (call_prices - y_true).pow(2)
        #vega_weights = torch.clamp(torch.abs(vega).detach(), min=self.vega_weight_eps)
        vega_weighted_mse = self.price_loss(call_prices, y_true)#(squared_error * vega_weights).sum() / vega_weights.sum()


        # Normalize theta to K
        theta_normalized = theta / torch.clamp(k, min=1e-8)

        # Calculate the theta floor based on Time-to-Maturity
        theta_floor = self.theta_floor_base - self.theta_floor_slope / torch.sqrt(
            torch.clamp(t, min=self.theta_floor_eps)
        )

        # Allow only negative floors this is for Call options
        theta_floor = torch.minimum(theta_floor, torch.zeros_like(theta_floor))

        delta_loss = torch.relu(-delta).pow(2).mean()
        delta_upper_loss = torch.relu(delta - self.delta_ceiling).pow(2).mean()
        dual_delta_loss = torch.relu(dual_delta).pow(2).mean()
        gamma_loss = torch.relu(-gamma).pow(2).mean()
        theta_loss = torch.relu(theta_floor - theta_normalized).pow(2).mean()
        theta_upper_loss = torch.relu(theta).pow(2).mean()
        dual_gamma_loss = torch.relu(-dual_gamma).pow(2).mean()
        vega_loss = torch.relu(-vega).pow(2).mean()

        # Call cannot cost more than the underlying asset
        price_upper_loss = torch.relu(call_prices - (s - self.price_spot_margin)).pow(2).mean()

        greek_penalty = (
            self.w_delta * delta_loss
            + self.w_delta_upper * delta_upper_loss
            + self.w_dual_delta * dual_delta_loss
            + self.w_gamma * gamma_loss
            + self.w_theta * theta_loss
            + self.w_theta_upper * theta_upper_loss
            + self.w_dual_gamma * dual_gamma_loss
            + self.w_price_upper * price_upper_loss
            + vega_loss
        )

        total_loss = vega_weighted_mse + self.lambda_arb * greek_penalty

        return {
            "loss": total_loss,
            "mse_loss": vega_weighted_mse,
            "greek_penalty": greek_penalty,
            "delta_loss": delta_loss,
            "delta_upper_loss": delta_upper_loss,
            "gamma_loss": gamma_loss,
            "theta_loss": theta_loss,
            "theta_upper_loss": theta_upper_loss,
            "dual_delta_loss": dual_delta_loss,
            "dual_gamma_loss": dual_gamma_loss,
            "price_upper_loss": price_upper_loss,
            "theta_floor_mean": theta_floor.mean(),
            "vega_loss": vega_loss,
            "vega_weighted_mse": vega_weighted_mse,
        }

class OptionNet(nn.Module):
    def __init__(self, n_inputs=3, n_hidden=64, n_layers=4):
        super().__init__()

        layers = []
        input_dims = n_inputs

        layers.append(
            nn.Linear(input_dims, n_hidden)
        )
        layers.append(nn.SiLU())

        layers.append(
            nn.Linear(n_hidden, n_hidden//2)
        )
        layers.append(nn.SiLU())

        layers.append(
            nn.Linear(n_hidden//2, n_hidden)
        )
        layers.append(nn.SiLU())

        layers.append(nn.Linear(n_hidden, 1))
        layers.append(nn.Softplus())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):

        xg = x.detach().requires_grad_(True)

        s = xg[:, 0:1]
        k = xg[:, 1:2]
        t = xg[:, 2:3]
        v = xg[:, 3:4]
        rest = xg[:, 4:]

        m = s / k

        model_input = torch.cat([m, t, v, rest], dim=1)

        normalized_prices = self.model(model_input)
        prices = normalized_prices * k

        delta, dual_delta, theta, vega = torch.autograd.grad(
            outputs=prices,
            inputs=[s, k, t, v],
            grad_outputs=torch.ones_like(prices),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )

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

        greeks = {
            "delta": delta,
            "dual_delta": dual_delta,
            "gamma": gamma,
            "dual_gamma": dual_gamma,
            "theta": -theta,
            "vega": vega,
        }

        return normalized_prices, greeks

class OptionNetModule(pl.LightningModule):
    def __init__(
            self,
            n_inputs=3,
            n_hidden=64,
            n_layers=4,
            lambda_arb = 1.0,
            theta_floor_base: float = -0.03,
            theta_floor_slope: float = 0.0393,
            theta_floor_eps: float = 1e-4,
            delta_ceiling: float = 0.9999,
            price_spot_margin: float = 1e-4,
            w_delta_upper: float = 1.0,
            w_theta_upper: float = 1.0,
            w_price_upper: float = 1.0,
            vega_weight_eps: float = 1e-8,
            learning_rate=1e-3,
            example_inputs = None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = OptionNet(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_layers=n_layers,
        )

        self.loss_fn = GreeksInformedLoss(
            lambda_arb=lambda_arb,
            theta_floor_base=theta_floor_base,
            theta_floor_slope=theta_floor_slope,
            theta_floor_eps=theta_floor_eps,
            delta_ceiling=delta_ceiling,
            price_spot_margin=price_spot_margin,
            w_delta_upper=w_delta_upper,
            w_theta_upper=w_theta_upper,
            w_price_upper=w_price_upper,
            vega_weight_eps=vega_weight_eps,
        )

        self.lambda_arb = lambda_arb
        self.example_input_array = example_inputs

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        y_pred = self(x)
        t = x[:, 2:3]
        s = x[:, 0:1]
        k = x[:, 1:2]
        loss_dict = self.loss_fn(y_pred, y, t=t, s=s, k=k)

        # Core logs
        self.log(f"{stage}_loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_mse_loss", loss_dict["mse_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_greeks_loss", self.lambda_arb * loss_dict["greek_penalty"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)

        # Component logs
        self.log(f"{stage}_delta_loss", loss_dict["delta_loss"], on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log(f"{stage}_delta_upper_loss", loss_dict["delta_upper_loss"], on_step=True, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f"{stage}_gamma_loss", loss_dict["gamma_loss"], on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log(f"{stage}_theta_loss", loss_dict["theta_loss"], on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log(f"{stage}_theta_upper_loss", loss_dict["theta_upper_loss"], on_step=True, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f"{stage}_dual_delta_loss", loss_dict["dual_delta_loss"], on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log(f"{stage}_dual_gamma_loss", loss_dict["dual_gamma_loss"], on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log(f"{stage}_price_upper_loss", loss_dict["price_upper_loss"], on_step=True, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f"{stage}_theta_floor_mean", loss_dict["theta_floor_mean"], on_step=True, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f"{stage}_vega_loss", loss_dict["vega_loss"], on_step=True, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f"{stage}_vega_weighted_mse", loss_dict["vega_weighted_mse"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)

        return loss_dict

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def on_validation_model_eval(self) -> None:
        super().on_validation_model_eval()
        torch.set_grad_enabled(True)

    def on_test_model_eval(self) -> None:
        super().on_test_model_eval()
        torch.set_grad_enabled(True)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="test")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }

        scheduler1 = StepLR(optimizer, step_size=10, gamma=0.95)

        return [optimizer], [scheduler1]
