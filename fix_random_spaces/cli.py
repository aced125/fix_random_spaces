"""Console script for fix_random_spaces."""
import sys
import click
import os
import logging
from transformers import BertTokenizer
import json
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from fix_random_spaces import utils
from fix_random_spaces.model import Model

logger = logging.getLogger(__name__)
os.environ["WANDB_API_KEY"] = "6beb9ef2d63f9b90456e658843c4e65ee59b88a9"


@click.group()
@click.option("-v", "--verbose", count=True)
def cli(verbose):
    """Console script for fix_random_spaces."""
    log_level = logging.INFO
    if verbose > 0:
        log_level = logging.DEBUG


@cli.command()
@click.argument("conf-file", type=click.Path(exists=True), default=None)
@click.option(
    "--hparams",
    default=json.dumps(
        {
            "name": "run2",
            "project": "seq2seq-clean-spaces",
            "train_bs": 16,
            "val_bs": 16,
            "num_workers": 4,
            "max_length": 160,
            "num_datapoints": 100_000,
            "optimizer": "Ranger",
            "optimizer_kwargs": {
                "lr": 3e-4,
                "alpha": 0.5,
                "betas": [0.95, 0.999],
                "eps": 1e-5,
                "weight_decay": 1e-3,
                # "use_gc": True,
            },
            "schedulers_kwargs": {"num_warmup_steps": 1000},
            "trainer_kwargs": {
                "gpus": 1,
                "gradient_clip_val": 0.5,
                "accumulate_grad_batches": 4,
                "min_epochs": 5,
                "max_epochs": 100,
                "precision": 32,
                "distributed_backend": None,
            },
        }
    ),
)
def train(conf_file, hparams):
    if conf_file is None:
        hparams = OmegaConf.create(hparams)
    else:
        hparams = OmegaConf.load(conf_file)
    print(hparams.pretty())

    log = WandbLogger(name=hparams.name, project=hparams.project)
    checkpoint = pl.callbacks.ModelCheckpoint(
        filepath="checkpoints", verbose=True, monitor="val_loss", mode="min"
    )
    trainer = pl.Trainer(
        logger=log, checkpoint_callback=checkpoint, **hparams.trainer_kwargs
    )
    model = Model(hparams)
    trainer.fit(model)


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
    # train()
