from typing import Tuple

import torch
import torch.nn as nn


def info_nce_loss(
    features: torch.Tensor, temperature: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the contrastive loss using the InfoNCE (Noise Contrastive Estimation) formulation.

    Args:
        features (torch.Tensor): Input tensor representing the features of the data.
        temperature (float, optional): Temperature parameter for scaling logits. Defaults to 0.05.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing logits and corresponding labels for the loss.

    References:
        [1] https://github.com/sthalles/SimCLR/blob/master/simclr.py

    Example:
        >>> features = torch.randn((64, 128))  # Example input features
        >>> logits, labels = info_nce_loss(features, temperature=0.1)
    """
    labels = torch.cat([torch.arange(len(features) // 2) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = nn.functional.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    return logits, labels
