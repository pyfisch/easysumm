import argparse
import glob
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from transformer_base import BaseTransformer, add_generic_args, generic_train, get_linear_schedule_with_warmup
from utils import SummarizationDataset
from transformers.tokenization_utils import trim_batch


logger = logging.getLogger(__name__)


class BART(BaseTransformer):
    mode = "language-modeling"

    def __init__(self, hparams):
        super().__init__(hparams, num_labels=None, mode=self.mode)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels,
        )

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,)

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask = trim_batch(
            batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"]
        )
        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_cnn.py
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=1,
            max_length=80,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )
        print(generated_ids)
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        loss = self._step(batch)

        return {"val_loss": loss, "preds": preds}

    def test_end(self, outputs):
        with open(self.hparams.pred_file, "w+") as p_writer:
            for output_batch in outputs:
                p_writer.writelines(s + "\n" for s in output_batch["preds"])
            p_writer.close()
        return self.validation_end(outputs)

    def get_dataloader(self, dataset_prefix: str, batch_size: int) -> DataLoader:
        dataset = SummarizationDataset(
            self.tokenizer,
            dataset_prefix,
            self.hparams.max_source_length,
            self.hparams.max_target_length
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader(self.hparams.train_prefix, batch_size=self.hparams.train_batch_size)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.hparams.valid_prefix, batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.hparams.test_prefix, batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        # Add BART specific options
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        return parser


def is_empty_dir(dir):
    return not os.path.exists(dir) or not os.listdir(dir)

def main(args):
    model = BART(args)
    trainer = generic_train(model, args)
    if args.do_train:
        trainer.fit(model)
        if args.model_dir is not None:
            os.makedirs(args.model_dir)
            model.model.save_pretrained(args.model_dir)
    if args.do_predict:
        trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = BART.add_model_specific_args(parser, os.getcwd())

    temp_args, _unknown = parser.parse_known_args()
    if temp_args.do_train:
        parser.add_argument("--train_prefix", type=str, required=True, help="The path file name prefix for the training data.")
        parser.add_argument("--valid_prefix", type=str, required=True, help="The path file name prefix for the validation data.")
        parser.add_argument("--checkpoint_dir", type=str, required=True, help="The directory where model checkpoints will be written.")
        parser.add_argument("--model_dir", type=str, help="The directory where the final model will be written.")
    if temp_args.do_predict:
        parser.add_argument("--test_prefix", type=str, required=True, help="The path file name prefix for the test data.")
        parser.add_argument("--pred_file", type=str, required=True, help="The file to which predictions are written.")

    args = parser.parse_args()

    if args.do_train and not is_empty_dir(args.checkpoint_dir):
        raise ValueError("The checkpoint directory ({}) exists and is not empty.".format(args.checkpoint_dir))
    if args.do_train and not is_empty_dir(args.model_dir):
        raise ValueError("The model directory ({}) exists and is not empty.".format(args.model_dir))
    if args.do_predict and os.path.exists(args.pred_file):
        raise ValueError("The predictions file ({}) already exists.".format(args.pred_file))

    main(args)
