import random
from logging import Logger
from itertools import product
from typing import Callable, Tuple, Set, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import (
    BaseAugmentation,
    AugmentationPipe,
    TruncateOrPad,
)
from .data_model import Patients, Patient, Diseases, Disease, HPOs


def collate_for_stochastic_pairwise_eval(x):
    sources, confirmed_diseases = tuple(zip(*x))
    sources = torch.stack(sources)
    return sources, confirmed_diseases


class ContrastiveDataset(Dataset):
    """Contrastive learning을 위한 데이터셋 생성

    Example:
        # 일반적인 데이터셋 생성
        >>> dataset = ContrastiveDataset(patients, diseases)
        >>> len(dataset)
        diseases * patients
        >>> patient_tensor, disease_tensor, label = dataset[0]

        # Augmentation
        >>> augmented_disease_hpo:HPOs = func(disease)
        >>> dataset_aug = ContrastiveDataset(patients, diseases, augmentator=func)
        >>> len(dataset_aug)
        aug_disease * patients

        # Padding
        >>> dataset_with_pad = ContrastiveDataset(patients, diseases, padding=True)
        >>> patient_tensor, disease_tensor, label = dataset[0]
        >>> patient_tensor.shape
        (250, 1735)
        >>> disease_tensor.shape

        >>> dataset[0]
        (
            tensor([[-0.0151, -0.0074,  0.0049,  ..., -0.0194, -0.0008, -0.0137],
            [-0.0244,  0.0020,  0.0214,  ..., -0.0044, -0.0333, -0.0264],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            ...,
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]
            ),
            tensor([[-0.0079,  0.0238,  0.0266,  ...,  0.0068, -0.0258, -0.0227],
                    [-0.0165,  0.0010,  0.0228,  ..., -0.0051,  0.0056, -0.0217],
                    [-0.0115, -0.0166,  0.0121,  ..., -0.0070, -0.0178, -0.0349],
                    ...,
                    [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]
            ),
            tensor([1.])
        )

    """

    def __init__(
        self,
        patients: Patients,
        diseases: Diseases,
        augmentator: Callable = None,
        padding: bool = False,
        device: str = "cpu",
        logger: Logger = Logger(__name__),
    ):
        self.patients = patients
        self.diseases = diseases
        self.augmentator = augmentator
        self.padding = padding
        self.device = device
        self.logger = logger
        self.__post_init__()

    def __post_init__(self):
        if self.padding:
            self.max_hpos = self._cal_max_padding()

        if self.augmentator is not None:
            augmentor_return_type = self.augmentator.__annotations__["return"]
            if augmentor_return_type != HPOs:
                raise TypeError(
                    "Augmentator's return type does not match HPOs, "
                    f"passed {augmentor_return_type}"
                )

        if self.augmentator is not None:
            self.diseases = Diseases(
                [self.augmentator(disease) for disease in self.diseases]
            )

        self.pairs = list(
            product(
                [patient.id for patient in self.patients],
                [disease.id for disease in self.diseases],
            )
        )

    def _cal_max_padding(self) -> int:
        max_hpos = 0
        for patient in self.patients:
            n_hpos = len(patient.hpos)
            if max_hpos > n_hpos:
                continue

            max_hpos = n_hpos

        for disease in self.diseases:
            n_hpos = len(disease.hpos)
            if max_hpos > n_hpos:
                continue

            max_hpos = n_hpos

        self.logger.debug("Set max padding: N=%s" % str(max_hpos))

        return max_hpos

    def __len__(self) -> int:
        return len(self.pairs)

    def _padding(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.ndim == 2

        n_hpos, n_dim = matrix.shape
        n_pad = self.max_hpos - n_hpos

        return np.concatenate([matrix, np.zeros(shape=(n_pad, n_dim))], axis=0)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patient_id, disease_id = self.pairs[index]

        patient: Patient = self.patients[patient_id]
        disease: Disease = self.diseases[disease_id]

        if disease.id in patient.disease_ids:
            label = np.array([1])
        else:
            label = np.array([-1])

        patient_hpos_vector: np.ndarray = patient.hpos.vector
        disease_hpos_vector: np.ndarray = disease.hpos.vector

        if self.augmentator:
            augmented_hpos: HPOs = self.augmentator(disease.hpos)
            disease_hpos_vector = augmented_hpos.vector

        if self.padding:
            patient_hpos_vector = self._padding(patient_hpos_vector)
            disease_hpos_vector = self._padding(disease_hpos_vector)

        return tuple(
            map(
                lambda x: torch.from_numpy(x).float().to(self.device),
                [
                    patient_hpos_vector,
                    disease_hpos_vector,
                    label,
                ],
            )
        )


class SamplingContrastiveDataset(ContrastiveDataset):
    """환자 표현형집합 - 질환표현형집합1개로 NxK개의 데이터셋 구축이 아닌,
    N x K' (K'은 샘플링)을 이용한 데이터셋

    Example:
        # Negative disease의 최대 갯수를 지정
        >>> sampling_dataset = SamplingContrastiveDataset(
                patients,
                diseases,
                n_disease=10,
                max_hpos=10,
                padding=True
            )
        >>> len(sampling_dataset)
        3100  # 환자수 * (원인질환 1개 + negative 질환 10개)

        # Iteration
        >>> for epoch in range(0, 10):
        >>>     sampling_dataset.suffle_pairs()
        >>>     trainer.run_epoch(epoch, sampling_dataset)

        # RankNet이용시 배치사이즈
        >>> n_disease = 10
        >>> sampling_dataset = SamplingContrastiveDataset(
                patients,
                diseases,
                n_disease=n_disease,
                max_hpos=10,
                padding=True
            )
        >>> dataloader = torch.utils.data.DataLoader(sampling_dataset, batch_size=n_disease)
        >>> for epoch in range(1, 10):
                trainer = Trainer(... loss=RankNet())
                sampling_dataset.suffle_pairs()
                trainer.run_epoch(epoch, sampling_dataset)
    """

    def __init__(
        self,
        patients: Patients,
        diseases: Diseases,
        n_disease: int = 10,
        max_hpos: int = -1,
        augmentator: Callable = None,
        padding: bool = False,
        device: str = "cpu",
        logger: Logger = Logger(__name__),
    ):
        self.patients = patients
        self.diseases = diseases
        self.augmentator = augmentator
        self.padding = padding
        self.device = device
        self.logger = logger
        self.n_disease = n_disease
        self.max_hpos = max_hpos
        self.__post_init__()
        self.suffle_pairs()

    def __post_init__(self) -> None:
        """초기화 후 후처리 함수"""

        if self.padding and self.max_hpos == -1:
            self.max_hpos = self._cal_max_padding()

        self._validate_augmentator()

        if self.augmentator is not None:
            self.diseases = Diseases(
                [self.augmentator(disease) for disease in self.diseases]
            )

    def _validate_augmentator(self):
        if self.augmentator is not None:
            augmentor_return_type = self.augmentator.__annotations__["return"]
            if augmentor_return_type != HPOs:
                raise TypeError(
                    "Augmentator's return type does not match HPOs, "
                    f"passed {augmentor_return_type}"
                )

    def suffle_pairs(self) -> None:
        """환자증상-질환증상의 랜덤으로 셔플링 + Negative sampling 수에 맞도록

        Note:
            다음의 로직으로 환자증상-질환증상 페어를 생성
            1) 환자증상-원인질환의 증상 (Positive pair)을 추가 K=1
            2) 전체질환에서 원인질환을 차집합하여 비원인질환을 선정
            3) 2)에서 구한 비원인질환에서 K-1개를 샘플링하여 총 N개의 페어를 생성
        """
        self.pairs = list()
        for patient in self.patients:
            positive_pair = [patient.id, random.sample(patient.disease_ids, k=1)[0]]
            self.pairs.append(positive_pair)

            negative_disease_ids: Set[str] = self.diseases.id2disease.keys() - set(
                patient.disease_ids
            )
            for disease_id in random.sample(negative_disease_ids, k=self.n_disease - 1):
                self.pairs.append([patient.id, disease_id])

        return

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patient_id, disease_id = self.pairs[index]

        patient: Patient = self.patients[patient_id]
        disease: Disease = self.diseases[disease_id]

        if disease.id in patient.disease_ids:
            label = np.array([1])
        else:
            label = np.array([-1])

        patient_hpos_vector: np.ndarray = patient.hpos.vector
        disease_hpos_vector: np.ndarray = disease.hpos.vector

        if len(disease_hpos_vector) > self.max_hpos:
            disease_hpos_vector = disease_hpos_vector[
                np.random.choice(len(disease_hpos_vector), size=self.max_hpos)
            ]

        if self.augmentator:
            augmented_hpos: HPOs = self.augmentator(disease.hpos)
            disease_hpos_vector = augmented_hpos.vector

        if self.padding:
            patient_hpos_vector = self._padding(patient_hpos_vector)
            disease_hpos_vector = self._padding(disease_hpos_vector)

        return tuple(
            map(
                lambda x: torch.from_numpy(x).float().to(self.device),
                [
                    patient_hpos_vector,
                    disease_hpos_vector,
                    label,
                ],
            )
        )


class StochasticPairwiseDataset(ContrastiveDataset):
    """
    Dataset for Stochastic Pairwise learning.

    Args:
        patients (Patients): Collection of patient data.
        diseases (Diseases): Collection of disease data.
        augmentation_factor (int, optional): Augmentation factor for training. Defaults to 30.
        max_len (int, optional): Maximum length of sequences. Defaults to 20.
        fraction_decay_rate (float, optional): Fraction decay rate for training. Defaults to 0.97.
        augmentator (List[BaseAugmentation], optional): List of augmentators. Defaults to None.
    """

    def __init__(
        self,
        patients: Patients,
        diseases: Diseases,
        augmentation_factor: int = 30,
        max_len: int = 20,
        initial_fraction: float = 0.75,
        fraction_decay_rate: float = 0.97,
        augmentator: List[BaseAugmentation] = None,
        logger: Logger = Logger(__name__),
    ):
        self.diseases, self.patients = diseases, patients
        self.max_len = max_len
        self.augmentation_factor = augmentation_factor
        self.fraction_decay_rate = fraction_decay_rate
        self._fraction = initial_fraction

        self.logger = logger
        self.augmentator = augmentator
        self.__post_init__()

    def __post_init__(self):
        self.padder = TruncateOrPad(
            self.max_len, stochastic=False, weighted_sampling=True
        )
        self._validate_augmentator()
        self.augment_pipe = AugmentationPipe(self.augmentator)
        self._assign_all_disease_tensors()
        self.all_disease_ids = self.diseases.all_disease_ids

    def _validate_augmentator(self):
        if self.augmentator is not None:
            for augmentator in self.augmentator:
                augmentor_return_type = augmentator.__call__.__annotations__["return"]
                if augmentor_return_type != torch.Tensor:
                    raise TypeError(
                        "Only support for return type of augmentator == 'torch.Tensor', "
                        f"passed {augmentor_return_type}"
                    )

    def train(self):
        """Switch to training mode."""
        self._train = True
        return self

    def validate(self):
        """Switch to validation mode."""
        self._train = False
        return self

    def step(self):
        """Decay the fraction for training."""
        self._fraction = max(self._fraction * self.fraction_decay_rate, 0.1)

    def _assign_all_disease_tensors(self):
        self.disease_tensors = {}
        for disease in self.diseases:
            disease_tensor = self.padder(
                torch.tensor(
                    disease.hpos.vector,
                    dtype=torch.float32,
                ),
                disease,
            )
            self.disease_tensors[disease.id] = disease_tensor

        self.all_symptom_vectors = torch.tensor(
            self.diseases.all_symptom_vectors, dtype=torch.float32
        )

    def __len__(self):
        if self._train:
            return len(self.diseases) * self.augmentation_factor
        else:
            return len(self.patients)

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Tuple containing positive pair (noise_sample, disease_sample) during training,
                   or (source, confirmed_disease) during validation.
        """

        if self._train:
            disease_id = list(self.all_disease_ids)[index // self.augmentation_factor]
            disease_sample = self.disease_tensors[disease_id]
            noise_sample = self.augment_pipe(disease_sample, fraction=self._fraction)
            positive_pair = (noise_sample, disease_sample)
            return positive_pair

        else:
            patient = self.patients[index]

            source = self.padder(
                torch.tensor(
                    patient.hpos.vector,
                    dtype=torch.float32,
                ),
                patient,
            )

            confirmed_disease = patient.disease_ids

            return source, confirmed_disease


class FinetuneDataset(StochasticPairwiseDataset):
    def __init__(
        self,
        patients,
        diseases,
        positive_ratio,
        use_synopsis: bool = False,
        *args,
        **kargs,
    ):
        super().__init__(patients, diseases, *args, **kargs)
        self.use_synopsis = use_synopsis
        self.positive_ratio = positive_ratio
        self._attach_synopsis()

    def __len__(self):
        return len(self.patients)

    def _attach_synopsis(self):
        if self.use_synopsis:
            for id_, vector in self.disease_tensors.items():
                disease = self.diseases[id_]
                if disease.clinical_synopsis:
                    synopsis_tensor = torch.tensor(
                        disease.clinical_synopsis.vector,
                        dtype=torch.float32,
                    )
                else:
                    synopsis_tensor = torch.zeros(
                        (vector.shape[-1],), dtype=torch.float32
                    )

                indicator_vector = -1 * torch.ones_like(
                    synopsis_tensor, dtype=torch.float32
                )
                synopsis_tensor += indicator_vector

                vector[0, :] = synopsis_tensor

                self.disease_tensors[id_] = vector

    def __getitem__(self, index):
        patient = self.patients[index]
        confirmed_disease = patient.disease_ids

        source = self.padder(
            torch.tensor(
                patient.hpos.vector,
                dtype=torch.float32,
            ),
            patient,
        )

        if np.random.rand() > 1 - self.positive_ratio:
            positive_disease_id = random.choice(list(confirmed_disease))
            positive_disease = self.disease_tensors[positive_disease_id]
            return (source, positive_disease), torch.tensor([1.0], dtype=torch.float32)

        else:
            negative_disease_id = random.choice(
                list(self.diseases.all_disease_ids - confirmed_disease)
            )
            negative_disease = self.disease_tensors[negative_disease_id]
            return (source, negative_disease), torch.tensor([-1.0], dtype=torch.float32)
