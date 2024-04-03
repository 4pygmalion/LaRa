import os
import sys
import warnings
import argparse

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# warnings.filterwarnings("ignore", category=FutureWarning)

import mlflow
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# from lightning.pytorch.loggers import MLFlowLogger

# from pytorch_lightning.callbacks import ModelCheckpoint
# import pytorch_lightning as pl
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
sys.path.append(ROOT_DIR)
from core.networks import Transformer
from core.io_ops import load_pickle
from core.data_model import Patients, Diseases, Disease
from core.datasets import (
    collate_for_stochastic_pairwise_eval,
    StochasticPairwiseDataset,
)
from core.augmentation import (
    cleanse_data,
    SampleSymptoms,
    AddNoiseSymptoms,
    TruncateOrPad,
)
from core.trainer import TransformerModelPretrain
from mlflow_settings import TRACKING_URI, EXP_SYMPTOM


torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Similarity Experiment")

    # mlflow run name
    parser.add_argument("--run_name", type=str, default="debug", help="Run Name")

    # dataset parameter
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="num workers for dataloader"
    )
    parser.add_argument("--num_devices", type=int, default=4, help="Number of Devices")
    parser.add_argument(
        "--augmentation_factor", type=int, default=100, help="Augmentation Factor"
    )
    parser.add_argument("--max_len", type=int, default=15, help="Augmentation Factor")
    parser.add_argument(
        "--initial_fraction",
        type=float,
        default=0.75,
        help="initial sampling fraction",
    )
    parser.add_argument(
        "--fraction_decay_rate",
        type=float,
        default=1.0,
        help="sampling fraction decay rate",
    )

    # model hyper params
    parser.add_argument(
        "--input_size", type=int, default=1536, help="Input ecoded vector size by LLM."
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Hidden dim for transformer encoder.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=128,
        help="Output dim of transformer encoder.",
    )
    parser.add_argument(
        "--nhead", type=int, default=16, help="nhead for transformer encoder layer."
    )
    parser.add_argument(
        "--n_layers", type=int, default=16, help="N layers for transformer encoder."
    )

    # optimizer params
    # parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument(
        "--val_interval",
        type=int,
        default=10,
        help="validation after every val_interval epochs",
    )
    parser.add_argument("--n_epoch", type=int, default=100, help="# of epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight Decay")

    # nce loss param
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature")

    # eval params
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[1, 5, 10, 100], help="ks for topk metric"
    )
    parser.add_argument("--random_state", type=int, default=2023)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    diseases: Diseases = load_pickle(os.path.join(DATA_DIR, "diseases.pickle"))
    omim_diseases = Diseases(
        [disease for disease in diseases if disease.id.startswith("OMIM:")]
    )
    omim_diseases = omim_diseases[omim_diseases.all_disease_ids]

    # all_symptom_vectors = torch.tensor(
    #     disease_data.all_symptom_vectors, dtype=torch.float32
    # )

    # mlf_logger = MLFlowLogger(
    #     experiment_name=EXP_SYMPTOM,
    #     run_name=args.run_name,
    #     tracking_uri=TRACKING_URI,
    # )

    # train_dataset = StochasticPairwiseDataset(
    #     train_patients,
    #     disease_data,
    #     augmentation_factor=args.augmentation_factor,
    #     max_len=args.max_len,
    #     initial_fraction=args.initial_fraction,
    #     fraction_decay_rate=args.fraction_decay_rate,
    #     augmentator=[
    #         SampleSymptoms(),
    #         AddNoiseSymptoms(all_symptom_vectors),
    #         TruncateOrPad(args.max_len, stochastic=True, weighted_sampling=False),
    #     ],
    # ).train()

    # val_dataset = StochasticPairwiseDataset(
    #     val_patients,
    #     disease_data,
    #     max_len=args.max_len,
    # ).validate()

    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )

    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     collate_fn=collate_for_stochastic_pairwise_eval,
    #     pin_memory=True,
    # )

    # trainer = pl.Trainer(
    #     max_epochs=args.n_epoch,
    #     devices=args.num_devices,
    #     accelerator="gpu",
    #     strategy="ddp_find_unused_parameters_false",
    #     check_val_every_n_epoch=args.val_interval,
    #     num_sanity_val_steps=0,
    #     logger=mlf_logger,
    # )

    # if trainer.is_global_zero:
    #     mlf_logger.log_hyperparams(vars(args))

    # # 0.3 × BatchSize/256
    # lr = args.batch_size * 0.3 / 256

    # model = TransformerModelPretrain(
    #     input_size=args.input_size,
    #     hidden_dim=args.hidden_dim,
    #     output_size=args.output_size,
    #     nhead=args.nhead,
    #     lr=lr,
    #     n_layers=args.n_layers,
    #     temperature=args.temperature,
    #     weight_decay=args.weight_decay,
    #     ks=args.ks,
    # )

    # trainer.fit(
    #     model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    # )

    # if trainer.is_global_zero:
    #     best_model = Transformer(**model.params)
    #     best_model.load_state_dict(torch.load(model.best_model_path))

    #     with mlflow.start_run(run_id=mlf_logger.run_id) as run:
    #         mlflow.pytorch.log_model(best_model, "model")
