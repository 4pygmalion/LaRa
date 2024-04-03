import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Callable


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from progress.bar import Bar
import pytorch_lightning as pl

# from flash.core.optimizers import LARS
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CORE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
sys.path.append(ROOT_DIR)
from core.loss import info_nce_loss
from core.networks import Transformer
from core_3asc.metric import AverageMeter, MetricHolder
from core_3asc.metric import topk_recall


@dataclass
class ResysMetrics(MetricHolder):
    loss: AverageMeter = AverageMeter()
    top1: AverageMeter = AverageMeter()
    top5: AverageMeter = AverageMeter()
    top10: AverageMeter = AverageMeter()
    top100: AverageMeter = AverageMeter()

    def reset(self):
        self.loss.reset()
        self.top1.reset()
        self.top5.reset()
        self.top10.reset()
        self.top100.reset()


class Trainer:
    def __init__(
        self, model, optimizer=None, loss: Callable = None, device: str = "cuda"
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.ks = [1, 5, 10, 100]

    def make_sentence(self, phase, epoch, step, total_steps, eta, **kwargs) -> str:
        base_sentence = (
            f"{phase} | EPOCH {epoch}: [{step}/{total_steps}] " f"| ETA: {eta} "
        )

        other_metrics = list()
        for metric_name, value in kwargs.items():
            other_metrics.append(f"{metric_name}: {value}")

        if not other_metrics:
            return base_sentence

        additoinal_sentence = "| ".join(other_metrics)
        return base_sentence + "| " + additoinal_sentence

    def _to_onehot(self, tensor: torch.Tensor) -> torch.Tensor:
        res = tensor.clone()
        mask = tensor != 1
        res[mask] = 0

        return res

    def run_train(self, epoch, dataloader) -> MetricHolder:
        self.model.train()

        metrics = ResysMetrics()
        metrics.reset()

        total_steps = len(dataloader)
        bar = Bar(max=total_steps)
        for step, batch in enumerate(dataloader, start=1):
            p_mat, d_mat, label = batch
            patient_repr, disease_repr = self.model(p_mat, d_mat)

            empirical_loss: torch.Tensor = self.loss(
                patient_repr, disease_repr, label.squeeze(dim=1)
            )

            self.optimizer.zero_grad()
            empirical_loss.backward()
            self.optimizer.step()

            metrics.update({"loss": empirical_loss.item()}, n=1)

            relevances = torch.nn.functional.cosine_similarity(
                patient_repr, disease_repr
            )

            onehot_label = self._to_onehot(label)
            relevances, labels = (
                relevances.detach().cpu().numpy(),
                onehot_label.detach().cpu().numpy(),
            )

            metrics.update(
                {f"top{k}": topk_recall(relevances, labels, k=k) for k in self.ks}, n=1
            )
            bar.suffix = self.make_sentence(
                "train",
                epoch,
                step=step,
                total_steps=total_steps,
                eta=bar.eta,
                **metrics.to_dict(prefix="train"),
            )
            bar.next()

        return metrics

    def run_eval(self, patients, diseases, epoch, phase) -> ResysMetrics:
        to_vec = (
            lambda x: torch.from_numpy(x.hpos.vector)
            .to(self.device)
            .unsqueeze(dim=0)
            .float()
        )

        if type(self.model) is torch.nn.DataParallel:
            model = self.model.module
        else:
            model = self.model

        disease_cache = dict()

        total_steps = len(patients)
        bar = Bar(max=total_steps)
        metrics = ResysMetrics()
        metrics.reset()
        for step, patient in enumerate(patients):
            with torch.no_grad():
                p_vector: torch.Tensor = model.represent(to_vec(patient))

            relevances = list()
            labels = list()
            for disease in diseases:
                if disease.id not in disease_cache:
                    with torch.no_grad():
                        disease_cache[disease.id] = model.represent(to_vec(disease))

                d_vector = disease_cache[disease.id]
                cosine_sim: float = torch.nn.functional.cosine_similarity(
                    p_vector, d_vector
                ).item()
                relevances.append(cosine_sim)

                label = 1 if disease.id in patient.disease_ids else 0
                labels.append(label)

                loss = 1 - cosine_sim if label else max(0, cosine_sim)

            metrics.update({"loss": loss}, n=1)
            relevances, labels = np.array(relevances), np.array(labels)
            metrics.update(
                {f"top{k}": topk_recall(relevances, labels, k=k) for k in self.ks}, n=1
            )
            bar.suffix = self.make_sentence(
                phase,
                epoch=epoch,
                step=step,
                total_steps=total_steps,
                eta=bar.eta,
                **metrics.to_dict(prefix=phase),
            )
            bar.next()

        return metrics


class TransformerModelPretrain(pl.LightningModule):
    _stage = "pretrain"

    def __init__(
        self,
        input_size,
        hidden_dim,
        output_size,
        nhead,
        n_layers,
        lr=1e-3,
        warmup_epoch=30,
        temperature=1.0,
        weight_decay=0.0,
        ks=[1, 5, 10, 100],
    ):
        """
        Transformer-based model with InfoNCE loss for representation learning.

        Args:
            input_size (int): Size of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_size (int): Size of the output features.
            nhead (int): Number of heads in the multiheadattention models.
            n_layers (int): Number of sub-encoder-layers in the transformer encoder.
            temperature (float): Temperature parameter for scaling logits in InfoNCE loss.
            weight_decay (float): L2 regularization term.
            lr (float): Learning rate.

        Attributes:
            inp_layer (nn.Linear): Input layer.
            tf_encode (nn.TransformerEncoder): Transformer encoder.
            last_layer (nn.Linear): Final output layer.
            temperature (float): Temperature parameter for InfoNCE loss.
            weight_decay (float): L2 regularization term.
            lr (float): Learning rate.
            info_nce_loss (function): InfoNCE loss function.
            criteria (nn.CrossEntropyLoss): Cross-entropy loss function.
            _cached_vector (dict): Cached vectors for validation.
            train_loss (AverageMeter): Average training loss.
            validation_step_outputs (list): Validation step outputs.

        Example:
            >>> model = TransformerModel(256, 512, 128, 8, 3, 0.07, 1e-4, 0.0003)
        """
        super(TransformerModelPretrain, self).__init__()
        self.params = {
            "input_size": input_size,
            "hidden_dim": hidden_dim,
            "output_size": output_size,
            "nhead": nhead,
            "n_layers": n_layers,
        }
        self.network = Transformer(**self.params)

        self.lr = lr
        self.warmup_epoch = warmup_epoch
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.ks = ks

        self.best_topk = 0.0

        self.info_nce_loss = info_nce_loss
        self.criteria = nn.CrossEntropyLoss()
        self._cached_vector = {}

        self.train_loss = AverageMeter()
        for k in self.ks:
            setattr(self, f"step_val_result_top_{k}", [])

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        batch = torch.cat(batch, dim=0)
        output_feat = self(batch)
        logits, labels = self.info_nce_loss(output_feat, self.temperature)
        loss = self.criteria(logits, labels)
        self.train_loss.update(loss.item())
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.is_global_zero and self.global_step % 30 == 0:
            self.logger.experiment.log_metric(
                self.logger.run_id,
                "train_loss",
                self.train_loss.val,
                step=self.global_step,
            )

    def on_train_epoch_end(self) -> None:
        self.train_loss.reset()
        if self.trainer.is_global_zero:
            self.logger.experiment.log_metric(
                self.logger.run_id,
                "sampling_fraction",
                self.trainer.train_dataloader.dataset.datasets._fraction,
                step=self.current_epoch,
            )
            self.logger.experiment.log_metric(
                self.logger.run_id,
                "current_learning_rate",
                self.lr_schedulers().get_last_lr()[-1],
                step=self.current_epoch,
            )

    def on_validation_epoch_start(self):
        self.eval()
        self._cached_vector = {}
        for k in self.ks:
            setattr(self, f"step_val_result_top_{k}", [])
        whole_disease = self.trainer.val_dataloaders[0].dataset.disease_tensors
        with torch.no_grad():
            with tqdm(
                total=len(whole_disease),
                position=0,
                leave=True,
                desc="Making all disease vectors...",
            ) as pbar:
                for id_, tensor in whole_disease.items():
                    self._cached_vector[id_] = self(tensor.to(self.device)).squeeze(0)
                    pbar.update(1)

    def validation_step(self, batch, batch_idx):
        input_src, confirmed_diseases = batch
        input_vector = self(input_src)
        whole_disease = self.trainer.val_dataloaders[0].dataset.disease_tensors
        for i, confirmed_disease in enumerate(confirmed_diseases):
            one_hot = np.zeros((len(whole_disease),))
            target_vectors = []
            for j, id_ in enumerate(whole_disease.keys()):
                target_vectors.append(self._cached_vector[id_])
                if id_ in confirmed_disease:
                    one_hot[j] = 1.0
            scores = (
                torch.nn.functional.cosine_similarity(
                    input_vector[[i]], torch.stack(target_vectors)
                )
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )
            for k in self.ks:
                getattr(self, f"step_val_result_top_{k}").append(
                    topk_recall(scores, one_hot, k=k)
                )

    def on_validation_epoch_end(self):
        for k in self.ks:
            gathered_outputs = self.all_gather(
                getattr(self, f"step_val_result_top_{k}")
            )

            if self.trainer.is_global_zero:
                if isinstance(gathered_outputs, list):
                    gathered_outputs = torch.cat(gathered_outputs)
                val_topk_recall = gathered_outputs.sum() / len(gathered_outputs)
                self.logger.experiment.log_metric(
                    self.logger.run_id,
                    f"val_top{k}_recall",
                    val_topk_recall,
                    step=self.current_epoch,
                )
        if self.trainer.is_global_zero:
            self.ckpt_dir = Path(DATA_DIR) / self._stage / self.logger.run_id
            if not self.ckpt_dir.exists():
                self.ckpt_dir.mkdir(parents=True)

            if val_topk_recall > self.best_topk:
                self.best_model_path = self.ckpt_dir / f"best_model.ckpt"
                with open(self.best_model_path, "wb") as f:
                    torch.save(self.network.state_dict(), f)
                self.best_topk = val_topk_recall

    def configure_optimizers(self):
        optimizer = LARS(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.warmup_epoch,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=self.lr,
        )
        return [optimizer], [scheduler]


class TransformerModelFinetune(TransformerModelPretrain):
    _stage = "finetune"

    def __init__(
        self, model, lr=1e-3, weight_decay=0.0, ks=[1, 5, 10, 100], margin=0.0
    ):
        super().__init__(**model.params)
        self.network = model

        self.lr = lr
        self.weight_decay = weight_decay
        self.ks = ks

        self.criteria = nn.CosineEmbeddingLoss(margin=margin)
        self._cached_vector = {}

        self.train_loss = AverageMeter()
        for k in self.ks:
            setattr(self, f"step_val_result_top_{k}", [])

    def training_step(self, batch, batch_idx):
        pair, label = batch
        pair = torch.cat(pair, dim=0)
        # (B*2)xp
        output_feat = self(pair)
        patient_embedding = output_feat[: len(pair) // 2]
        disease_embedding = output_feat[len(pair) // 2 :]
        loss = self.criteria(patient_embedding, disease_embedding, label.squeeze(-1))
        self.train_loss.update(loss.item())
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.is_global_zero:
            self.logger.experiment.log_metric(
                self.logger.run_id,
                "train_loss",
                self.train_loss.val,
                step=self.global_step,
            )

    def on_train_epoch_end(self) -> None:
        self.train_loss.reset()
        if self.trainer.is_global_zero:
            self.logger.experiment.log_metric(
                self.logger.run_id,
                "current_learning_rate",
                self.lr_schedulers().get_last_lr()[-1],
                step=self.current_epoch,
            )
