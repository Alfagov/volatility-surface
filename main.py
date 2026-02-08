import warnings
from datetime import datetime
from pathlib import Path

import torch

from dataset import OptionsDataModule
from model import OptionNetModule
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    lr = 7e-4
    EPOCHS = 100
    batch_size = 1024
    LAMBDA_ARB = 2.0

    torch.manual_seed(42)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    checkpoint_dir = Path("./checkpoints") / f"run_{run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch={epoch:02d}-step={step}",
        save_last=True,
        auto_insert_metric_name=False,
    )

    import mlflow
    from lightning.pytorch.loggers import MLFlowLogger

    from lightning.pytorch.callbacks import LearningRateMonitor

    data_module = OptionsDataModule("./data/108105", sofr_path="./data/sofr.csv", batch_size=batch_size)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = OptionNetModule(
        n_inputs=9,
        n_hidden=64,
        n_layers=2,
        lambda_arb=LAMBDA_ARB,
        theta_floor_base=-0.03,
        theta_floor_slope=0.0393,
        learning_rate=lr,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        logger=MLFlowLogger(experiment_name="surfaces", tracking_uri="file:./ml-runs", log_model="all"),
        callbacks=[lr_monitor, checkpoint_callback],
        inference_mode=False,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
