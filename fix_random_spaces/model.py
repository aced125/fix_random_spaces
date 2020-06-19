import pytorch_lightning as pl
from transformers import EncoderDecoderModel, BertTokenizer
import torch
import torch_optimizer
import torch.utils.data as tud
from fix_random_spaces import utils


class NoamScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, num_warmup_steps=1000, last_epoch=-1):
        assert num_warmup_steps > 0
        normalize = 1 / (num_warmup_steps * num_warmup_steps ** -1.5)
        super().__init__(
            optimizer,
            lambda step: normalize
            * min((step + 1) ** -0.5, (step + 1) * num_warmup_steps ** -1.5),
            last_epoch,
        )


class Model(pl.LightningModule):
    def __init__(
        self, hparams, train_dataset=None, val_dataset=None, test_dataset=None
    ):
        super().__init__()
        self.hparams = hparams
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            "bert-base-cased", "bert-base-cased"
        )  # initialize Bert2Bert
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.collater = utils.Collater(self.tokenizer, self.hparams.max_length)
        # self.train_dataset = train_dataset
        # self.val_dataset = val_dataset
        # self.test_dataset = test_dataset

    def prepare_data(self) -> None:
        self.train_dataset = utils.prepare_dataset(
            self.tokenizer,
            "train",
            self.hparams.max_length,
            self.hparams.num_datapoints,
        )
        self.val_dataset = utils.prepare_dataset(
            self.tokenizer,
            "validation",
            self.hparams.max_length,
            self.hparams.num_datapoints,
        )
        self.test_dataset = utils.prepare_dataset(
            self.tokenizer, "test", self.hparams.max_length, self.hparams.num_datapoints
        )

    def train_dataloader(self):
        return tud.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.train_bs,
            shuffle=True,
            num_workers=self.hparams.num_workers or 4,
            collate_fn=self.collater,
        )

    def val_dataloader(self):
        return tud.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_bs,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collater,
        )

    def test_dataloader(self):
        return tud.DataLoader(
            self.test_dataset,
            self.hparams.val_bs,
            False,
            num_workers=self.hparams.num_workers or 4,
            collate_fn=self.collater,
        )

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        loss, logits, *_ = self(batch)
        self.logger.log_metrics({"loss": loss.cpu()})
        output = {"loss": loss}
        return output

    def validation_step(self, batch, batch_idx):
        return self._shared_val_step(batch, batch_idx, "val")

    def validation_epoch_end(self, output):
        return self._shared_val_end(output, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_val_step(batch, batch_idx, "test")

    def test_epoch_end(self, output):
        return self._shared_val_end(output, "test")

    def _shared_val_step(self, batch, batch_idx, prefix):
        loss, logits, *_ = self(batch)
        preds = logits.argmax(-1)  # bs x seqlen
        lm_labels = batch["lm_labels"]  # bs x seqlen

        acc_mask = lm_labels[:, 1:].ne(-100)
        correct = preds[:, :-1].eq(lm_labels[:, 1:])  # bs x (seqlen - 1)
        frac_tokens_correct = correct.masked_select(acc_mask).float().mean()
        correct[~acc_mask] = True
        frac_seqs_correct = correct.all(1).float().mean()

        logs = {
            f"{prefix}_loss": loss,
            "frac_tokens_correct": frac_tokens_correct,
            "frac_seqs_correct": frac_seqs_correct,
        }
        return logs

    def _shared_val_end(self, output, prefix):
        output = self.collate(output)
        logs = {"log": output, f"{prefix}_loss": output[f"{prefix}_loss"]}
        # self.logger.log_metrics(output)
        return logs

    def configure_optimizers(self):
        opt_class = getattr(torch_optimizer, self.hparams.optimizer)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.optimizer_kwargs.weight_decay or 1e-7,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = opt_class(
            optimizer_grouped_parameters, **self.hparams.optimizer_kwargs
        )
        scheduler = NoamScheduler(
            self.optimizer, self.hparams.schedulers_kwargs.num_warmup_steps
        )
        self.scheduler = {"scheduler": scheduler, "interval": "step"}
        return [self.optimizer], [self.scheduler]

    def collate(self, output):
        keys = output[0].keys()
        return_dict = {}
        for key in keys:
            tensor = output[0][key]
            if tensor.dim() == 0:
                return_dict[key] = torch.stack([x[key] for x in output]).mean()
            elif tensor.dim() == 1:
                return_dict[key] = torch.cat([x[key] for x in output]).mean()
        return return_dict
