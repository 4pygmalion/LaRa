"""
$ python3 train_ssl.py \
    --run_name testing \
    --batch_size 16 \
    --device "cuda:3"
"""

import os
import sys
import math
import copy
import argparse

import mlflow
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
sys.path.append(ROOT_DIR)
from core.networks import Transformer
from core.io_ops import load_pickle
from core.data_model import  Diseases
from core.datasets import (
    DiseaseSSLDataSet,
)
from core.transforms import TruncateOrPad
from core.augmentation import (
    SampleSymptoms,
    AddNoiseSymptoms,
)
from SimCLR.metrics import AverageMeter
from SimCLR.trainer import SimCLRTrainer
from SimCLR.loss import SimCLRLoss
from mlflow_settings import get_experiment

sys.setrecursionlimit(50000)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def parse_args():
    parser = argparse.ArgumentParser(description="LaRA Self-supervised learning")
    parser.add_argument("--run_name", type=str, default="debug", help="Run Name")

    # dataset parameter
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="num workers for dataloader"
    )
    parser.add_argument("--num_devices", type=int, default=4, help="Number of Devices")
    parser.add_argument("--max_len", type=int, default=30, help="Number of HPOs in a item")
    parser.add_argument("--fraction", type=float, default=0.8, help="Sampling ratio")
    parser.add_argument(
        "--input_size", type=int, default=1536, help="Input ecoded vector size by LLM."
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=2048,
        help="Hidden dim for transformer encoder.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=512,
        help="Output dim of transformer encoder.",
    )
    parser.add_argument(
        "--nhead", type=int, default=32, help="nhead for transformer encoder layer."
    )
    parser.add_argument(
        "--n_layers", type=int, default=32, help="N layers for transformer encoder."
    )
    parser.add_argument("--n_epoch", type=int, default=100, help="# of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_patience", type=float, default=7, help="patience for early stopping")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight Decay")

    # nce loss param
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")

    # eval params
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[1, 5, 10, 100], help="ks for topk metric"
    )
    parser.add_argument("--random_state", type=int, default=2023)
    parser.add_argument("--device", help=str, default='cuda')

    return parser.parse_args()


def build_dataloader(omim_diseases, max_len, batch_size, num_workers, fraction, device) -> torch.utils.data.DataLoader:
    """SSL용 데이터로더를 추가합니다."""
                
    dataset = DiseaseSSLDataSet(
        omim_diseases,
        augmentators=[
            SampleSymptoms(fraction),
            SampleSymptoms(fraction),
            AddNoiseSymptoms(omim_diseases.all_symptom_vectors)
        ],
        transforms=TruncateOrPad(max_len),
        device=device
    )
        
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    args = parse_args()
    
    diseases: Diseases = load_pickle(os.path.join(DATA_DIR, "diseases.pickle"))
    omim_diseases = Diseases(
        [disease for disease in diseases if disease.id.startswith("OMIM:")]
    )
    
    # DataLoader
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_len": args.max_len,
        "fraction": args.fraction,
        "device": args.device
    }
    train_dataloader = build_dataloader(omim_diseases, **loader_args)
    val_dataloader = build_dataloader(omim_diseases, **loader_args)
    test_dataloder = build_dataloader(omim_diseases, **loader_args)
    
    model = Transformer(
        input_size=args.input_size,
        hidden_dim=args.hidden_dim,
        output_size=args.output_size,
        nhead=args.nhead,
        n_layers=args.n_layers,
    ).to(args.device)
    
    loss = SimCLRLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    trainer = SimCLRTrainer(
        model, 
        loss, 
        optimizer, 
        device=args.device
    )
    
    mlflow_exp = get_experiment()
    best_params = dict()
    patience = 0
    best_loss = math.inf
    with mlflow.start_run(experiment_id=mlflow_exp.experiment_id, run_name=args.run_name):
        mlflow.log_artifact(os.path.abspath(__file__))
        mlflow.log_params(vars(args))
        
        for epoch in range(1, args.n_epoch+1):
            train_loss_meter, train_metric_meter = trainer.run_epoch(
                train_dataloader,
                phase="train",
                epoch=epoch
            )
            val_loss_meter, val_metric_meter = trainer.run_epoch(
                val_dataloader,
                phase="val",
                epoch=epoch,
            )
            
            mlflow.log_metric("train_loss", train_loss_meter.avg, step=epoch)
            mlflow.log_metric("train_top_5", train_metric_meter.avg, step=epoch)
            mlflow.log_metric("val_loss", val_loss_meter.avg, step=epoch)
            mlflow.log_metric("val_top_5", val_metric_meter.avg, step=epoch)
            if best_loss >= val_loss_meter.avg:
                best_loss = val_loss_meter.avg
                patience = 0
                best_params:dict = copy.deepcopy(model.state_dict())
                continue
            
            if patience == args.max_patience:
                break
            
            patience += 1

        model.load_state_dict(best_params)
        test_loss_meter, test_metric_meter = trainer.run_epoch(
            test_dataloder,
            phase="test",
            epoch=0,
        )
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metric("test_loss", test_loss_meter.avg, step=epoch)
        mlflow.log_metric("test_top_5", test_metric_meter.avg, step=epoch)
            
            