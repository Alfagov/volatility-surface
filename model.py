from typing import Any, Tuple, Dict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import lightning as pl
from torch.optim import Adam


class GreeksInformedLoss(nn.Module):
    def __init__(
            self,
            lambda_arb = 1.0,
            theta_floor_base: float = -0.03,
            theta_floor_slope: float = 0.0393,  # tuned on training data: ~5-10th pct of theoretical theta/K by maturity
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
    ):
        super().__init__()
        self.mse = nn.MSELoss()
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

    def forward(
            self,
            y_pred: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
            y_true: torch.Tensor,
            t: torch.Tensor, # maturity from input batch, [B, 1]
            s: torch.Tensor,
            k: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        pred, greeks = y_pred
        mse_loss = self.mse(pred, y_true)

        delta = greeks["delta"]
        dual_delta = greeks["dual_delta"]
        gamma = greeks["gamma"]
        theta = greeks["theta"]
        dual_gamma = greeks["dual_gamma"]
        call_prices = pred * k

        # Normalize theta by strike so floor parameters are in normalized-price units.
        theta_normalized = theta / torch.clamp(k, min=1e-8)
        # Nonlinear floor: very negative near expiry, relaxing toward base at long maturities.
        theta_floor = self.theta_floor_base - self.theta_floor_slope / torch.sqrt(
            torch.clamp(t, min=self.theta_floor_eps)
        )
        # Keep floor non-positive to remain consistent with call-theta calendar monotonicity (theta <= 0).
        theta_floor = torch.minimum(theta_floor, torch.zeros_like(theta_floor))

        delta_loss = torch.relu(-delta).pow(2).mean()
        delta_upper_loss = torch.relu(delta - self.delta_ceiling).pow(2).mean()
        dual_delta_loss = torch.relu(dual_delta).pow(2).mean()
        gamma_loss = torch.relu(-gamma).pow(2).mean()
        theta_loss = torch.relu(theta_floor - theta_normalized).pow(2).mean()
        theta_upper_loss = torch.relu(theta).pow(2).mean()
        dual_gamma_loss = torch.relu(-dual_gamma).pow(2).mean()
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
        )

        total_loss = mse_loss + self.lambda_arb * greek_penalty

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
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
        layers.append(nn.Softplus())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):

        xg = x.detach().requires_grad_(True)

        s = xg[:, 0:1]
        k = xg[:, 1:2]
        t = xg[:, 2:3]
        rest = xg[:, 3:]

        m = s / k

        model_input = torch.cat([m, t, rest], dim=1)

        normalized_prices = self.model(model_input)
        prices = normalized_prices * k

        delta, dual_delta, theta = torch.autograd.grad(
            outputs=prices,
            inputs=[s, k, t],
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

        scheduler1 = StepLR(optimizer, step_size=6, gamma=0.9)

        return [optimizer], [scheduler1]
