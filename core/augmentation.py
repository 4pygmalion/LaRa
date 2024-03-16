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


def cleanse_data(diseases, patients):
    """
    질병 데이터와 환자 데이터 중 쓸 수 없는 데이터를 아래의 기준으로 필터링

    - 질병 데이터
        - OMIM이 아닌(ORPHA)등의 id를 가진 증상을 하나라도 가질 경우 

    - 환자 데이터
        - 아래의 네가지 조건을 기준으로 필터링

        - zero_symptom: 기록된 증상이 하나도 없는 경우
        - too_many_symptom: 기록된 증상이 50개를 초과하는 경우 (가끔 매우 많이 들어오는 환자가 있어요)
        - disease_id_missing: [HPO_VERSION = "v2023-10-09"]에서 포함되지 않는 증상이 있는 경우
    """
    not_omim = lambda x: not x.id.startswith("OMIM")
    filtered_disease = []
    for d_data in diseases:
        conditions = [not_omim(d_data)]
        if any(conditions):
            # print(f"{d_data} filtered due to conditions: {conditions}")
            continue
        else:
            filtered_disease.append(d_data)

    diseases = Diseases(filtered_disease)
    # remove duplication
    diseases = diseases[diseases.all_disease_ids]

    zero_symptom = lambda x: len(x.hpos) == 0
    too_many_symptom = lambda x: len(x.hpos) > 50
    disease_id_missing = lambda x: "-" in x.disease_ids
    not_in_hpo_db = lambda x: any(
        [id_ not in diseases.all_disease_ids for id_ in list(x.disease_ids)]
    )

    filtered_patients = []
    for p_data in patients:
        conditions = [
            zero_symptom(p_data),
            too_many_symptom(p_data),
            disease_id_missing(p_data),
            not_in_hpo_db(p_data),
        ]
        if any(conditions):
            # print(f"{p_data} filtered due to conditions: {conditions}")
            continue
        else:
            filtered_patients.append(p_data)

    patients = Patients(filtered_patients)

    return diseases, patients


class BaseAugmentation:
    def __call__(self):
        pass


class TruncateOrPad(BaseAugmentation):
    def __init__(
        self, max_len: int, stochastic: bool = True, weighted_sampling: bool = True
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.stochastic = stochastic
        self.weighted_sampling = weighted_sampling

    def __call__(
        self,
        symptom_seq: torch.Tensor,
        symptom_set: Union[Patient, Disease] = None,
    ) -> torch.Tensor:
        if symptom_set is None and self.weighted_sampling:
            if symptom_set is None:
                raise ValueError(
                    "Symptom set is missing, weight for sampling can not be calcalated!"
                )

        len_diff = self.max_len - len(symptom_seq)
        if len_diff < 0:
            if self.weighted_sampling:
                p = np.array([hpo.depth for hpo in symptom_set.hpos])
                p = p / p.sum()
            else:
                p = None
            sample_idx = (
                np.random.choice(
                    range(len(symptom_seq)), size=self.max_len, replace=False, p=p
                )
                if self.stochastic
                else np.argsort(p)[-self.max_len :]
            )

            symptom_seq = symptom_seq[sample_idx, :]

        elif len_diff > 0:
            symptom_seq = torch.nn.functional.pad(
                symptom_seq, (0, 0, 0, len_diff), mode="constant", value=0
            )

        return symptom_seq


class SampleSymptoms(BaseAugmentation):
    def __call__(
        self, symptom_seq: torch.Tensor, fraction: float = 0.75
    ) -> torch.Tensor:
        n = len(symptom_seq)
        n_syms = np.clip(np.random.poisson(int(n * fraction)), 1, n)
        sample_idx = np.random.choice(range(n), size=n_syms, replace=False)
        return symptom_seq[sample_idx]


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
