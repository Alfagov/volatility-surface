import warnings
import torch

from dataset import OptionsDataModule
from model import OptionNetModule
import lightning as pl
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    lr = 1e-6
    EPOCHS = 1
    batch_size = 128
    LAMBDA_ARB = 10.0

    torch.manual_seed(42)

    from lightning.pytorch.loggers import LitLogger, TensorBoardLogger

    data_module = OptionsDataModule("./data/108105", batch_size=batch_size)

    model = OptionNetModule(
        n_inputs=3,
        n_hidden=128,
        n_layers=2,
        lambda_arb=LAMBDA_ARB,
        learning_rate=lr
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
        logger=TensorBoardLogger("./logs/", name="my_experiment")
    )

    trainer.fit(model, data_module)

    # Visualization
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    val_loader = data_module.val_dataloader()

    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            preds, _ = model(x)
            all_preds.append(preds.numpy())
            all_actuals.append(y.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_actuals = np.concatenate(all_actuals).flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(all_actuals, all_preds, alpha=0.5)
    plt.plot([all_actuals.min(), all_actuals.max()], [all_actuals.min(), all_actuals.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Predicted vs Actual Option Prices')
    plt.savefig('predicted_vs_actual.png')
    print("Scatterplot saved to predicted_vs_actual.png")