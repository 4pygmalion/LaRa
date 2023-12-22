"""
$ python3 train_model.py \
    --run_name RDScanner \
    --n_epoch 100 \
    --hidden_dim 512 \
    --out_dim 512

"""
import os
import math
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import mlflow
import torch
import numpy as np

from sklearn.model_selection import train_test_split

from ontology_src import ARTIFACT_PATH
from mlflow_settings import TRACKING_URI, EXP_SYMPTOM
from core.io_ops import load_pickle
from core.data_model import Diseases, Patients
from core.datasets import SamplingContrastiveDataset
from core.networks import RDScanner
from core.trainer import Trainer
from log_ops import get_logger


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name", help="mlflow run name", type=str, default="default"
    )
    parser.add_argument("--maximal_hpo", type=int, default=1000)
    parser.add_argument("--n_disease", type=int, default=30)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=2023)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--max_patience", type=int, default=5)

    return parser.parse_args()


def metric_to_sentence(metrics, phase):
    sentence = " | ".join(
        [f"{metric}:{value}" for metric, value in metrics.to_dict(phase).items()]
    )

    return sentence


if __name__ == "__main__":
    ARGS = get_args()
    LOGGER = get_logger("train_model")
    LOGGER.info("---ARGS----")
    for name, value in vars(ARGS).items():
        LOGGER.info("ARGS(%s): %s" % (name, value))

    ## MLFLOW
    mlflow.set_tracking_uri(TRACKING_URI)
    MLFLOW_CLIENT = mlflow.MlflowClient(tracking_uri=TRACKING_URI)
    exp = MLFLOW_CLIENT.get_experiment_by_name(EXP_SYMPTOM)
    if exp is None:
        exp = MLFLOW_CLIENT.create_experiment(MLFLOW_CLIENT)
    EXP_ID = exp.experiment_id
    RUN_NAME = ARGS.run_name

    ## FILTERING
    patients = load_pickle(ARTIFACT_PATH["patients"])
    diseases = load_pickle(ARTIFACT_PATH["diseases"])
    omim_disease = Diseases(
        [disease for disease in diseases if disease.id.startswith("OMIM")]
    )

    train_val_patients_list, test_patients_list = train_test_split(
        patients.data, random_state=ARGS.random_state
    )
    train_patients_list, val_patients_list = train_test_split(
        train_val_patients_list, random_state=ARGS.random_state
    )
    train_patients = Patients(train_patients_list)
    val_patients = Patients(val_patients_list)
    test_patients = Patients(test_patients_list)

    with mlflow.start_run(run_name=RUN_NAME, experiment_id=EXP_ID) as parent_run:
        mlflow.log_params(vars(ARGS))
        mlflow.log_artifact(os.path.abspath(__file__))
        mlflow_data = mlflow.data.from_numpy(np.array(patients.data))
        mlflow.log_input(mlflow_data, context="entire")
        mlflow.log_input(
            mlflow.data.from_numpy(np.array(test_patients.data)), context="test"
        )

        train_dataset = SamplingContrastiveDataset(
            train_patients,
            omim_disease,
            n_disease=ARGS.n_disease,
            device="cuda",
            padding=True,
        )
        train_dataloder = torch.utils.data.DataLoader(
            train_dataset, batch_size=ARGS.n_disease
        )

        model = RDScanner(hidden_dim=ARGS.hidden_dim, out_dim=ARGS.out_dim).to("cuda")
        dd_model = torch.nn.DataParallel(model)

        trainer = Trainer(
            dd_model,
            optimizer=torch.optim.Adam(dd_model.parameters()),
            loss=torch.nn.functional.cosine_embedding_loss,
            device="cuda",
        )
        patience = 0
        best_loss = math.inf
        for epoch in range(1, ARGS.n_epoch + 1):
            train_dataset.suffle_pairs()

            train_metrics = trainer.run_train(epoch=epoch, dataloader=train_dataloder)
            mlflow.log_metrics(train_metrics.to_dict(prefix="train"), step=epoch)
            LOGGER.info("Train: " + metric_to_sentence(train_metrics, phase="train"))

            val_metrics = trainer.run_eval(
                val_patients, omim_disease, epoch=epoch, phase="val"
            )
            mlflow.log_metrics(val_metrics.to_dict(prefix="val"), step=epoch)
            LOGGER.info("Val: " + metric_to_sentence(val_metrics, phase="val"))

            val_loss = val_metrics.loss.avg
            if best_loss > val_loss:
                best_loss = val_loss
                patience = 0

            else:
                patience += 1
                if patience == ARGS.max_patience:
                    break

        torch.save(dd_model.state_dict(), "dd_model.pth")
        torch.save(model.state_dict(), "model.pth")
        test_metrics = trainer.run_eval(
            test_patients, omim_disease, epoch=1, phase="test"
        )
        mlflow.log_metrics(test_metrics.to_dict(prefix="test"))
        LOGGER.info("Test: " + metric_to_sentence(test_metrics, phase="test"))

        mlflow.pytorch.log_model(model, artifact_path="model")
