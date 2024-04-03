from abc import abstractmethod, ABC
from typing import Any, Union, List

import numpy as np
import torch

import os
import sys

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CORE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
sys.path.append(ROOT_DIR)

from core.data_model import Patients, Patient, Diseases, Disease


class BaseAugmentation(ABC):
    @abstractmethod
    def __call__(self):
        """Abstract method must be implemented"""
        pass


class SampleSymptoms(BaseAugmentation):
    
    def __init__(self, fraction:float=0.75):
        self.fraction = fraction
        
    def __call__(self, hpo_tensors: torch.Tensor) -> torch.Tensor:
        n = len(hpo_tensors)
        n_syms = np.clip(np.random.poisson(int(n * self.fraction)), 1, n)
        sample_idx = np.random.choice(range(n), size=n_syms, replace=False)
        return hpo_tensors[sample_idx]


class AddNoiseSymptoms(BaseAugmentation):
    def __init__(self, all_symptom_vectors) -> None:
        super().__init__()
        self.all_symptom_vectors = all_symptom_vectors

    def __call__(
        self,
        symptom_seq: torch.Tensor,
    ) -> torch.Tensor:
        n_noise = np.clip(np.random.poisson(1), 0, 2)
        sample_idx = np.random.choice(
            len(self.all_symptom_vectors), size=n_noise, replace=False
        )
        noise_vector = self.all_symptom_vectors[sample_idx]
        return torch.cat([symptom_seq, noise_vector])


class AugmentationPipe:
    """
    Pipeline for applying a sequence of augmentations to symptom sequences.

    Args:
        augmentations (List[BaseAugmentation]): List of augmentation objects to apply.

    Attributes:
        ARGUMENTS (dict): Dictionary mapping augmentation class names to their required arguments.
        augmentations (List[BaseAugmentation]): List of augmentation objects to apply.

    Methods:
        __call__(
            symptom_seq: torch.Tensor,
            symptom_set: Union[Patient, Disease] = None,
            fraction: float = 0.75,
        ) -> torch.Tensor:
            Apply a sequence of augmentations to a symptom sequence.

    Example:
        all_symptom_vectors = torch.tensor(
            disease_data.all_symptom_vectors, dtype=torch.float32
        )
        augmentation_pipe = AugmentationPipe(
            [
                SampleSymptoms(),
                AddNoiseSymptoms(all_symptom_vectors),
                TruncateOrPad(max_len=15, stochastic=True, weighted_sampling=True),
            ]
        )
        sample = disease_data[0]
        sample_vec = torch.tensor(disease_data[0].hpos.vector)
        augmented_sample = augmentation_pipe(sample_vec)
    """

    ARGUMENTS = {
        "TruncateOrPad": ("symptom_seq", "symptom_set"),
        "SampleSymptoms": ("symptom_seq", "fraction"),
        "AddNoiseSymptoms": ("symptom_seq",),
    }

    def __init__(self, augmentations: List[BaseAugmentation]) -> None:
        """
        Initialize the AugmentationPipe.

        Args:
            augmentations (List[BaseAugmentation]): List of augmentation objects to apply.
        """
        self.augmentations = augmentations

    def __call__(
        self,
        symptom_seq: torch.Tensor,
        symptom_set: Union[Patient, Disease] = None,
        fraction: float = 0.75,
    ) -> torch.Tensor:
        """
        Apply a sequence of augmentations to a symptom sequence.

        Args:
            symptom_seq (torch.Tensor): Input symptom sequence.
            symptom_set (Union[Patient, Disease], optional): Symptom set associated with the sequence.
            fraction (float, optional): Fraction used in SampleSymptoms augmentation.

        Returns:
            torch.Tensor: Augmented symptom sequence.
        """
        self.symptom_seq = symptom_seq
        self.symptom_set = symptom_set
        self.fraction = fraction

        for augmentation in self.augmentations:
            argument_names = self.ARGUMENTS.get(augmentation.__class__.__name__, ())
            arguments = [getattr(self, arg_name) for arg_name in argument_names]
            self.symptom_seq = augmentation(*arguments)

        return self.symptom_seq
