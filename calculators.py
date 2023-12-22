import logging

from abc import ABCMeta, abstractmethod
from typing import Dict, Set

from core.data_model import HPO, Ontology
from core.io_ops import load_pickle
from ontology_src import ARTIFACT_PATH


class BaseSimilarityCalculator(metaclass=ABCMeta):
    """증상유사도 계산기의 추상화클래스"""

    def __init__(
        self,
        logger=logging.Logger(__name__),
    ) -> None:
        self.ontology: Ontology = Ontology(load_pickle(ARTIFACT_PATH["hpo_definition"]))
        self.logger = logger

        return

    # TODO: Known disease(HPOs)의 정보를 저장
    def set_disease_phenotype_map(self):
        self.disease_pheno_map = dict()
        return

    @abstractmethod
    def cal_similarity_two_hpos(self, hpo_q: HPO, hpo_d: HPO) -> float:
        """두 HPO사이의 의미론적 유사도 계산"""
        raise NotImplementedError

    @abstractmethod
    def get_semantic_similarity_one_side(
        self, phenotypes1: Set[HPO], phenotypes2: Set[HPO]
    ):
        """두 집합 개념(phenotypes, phenotypes)의 ontology의 유사성을 단방향
        Q(phenotypes, 쿼리)-> D(phenotypes, 질환)으로 유사성을 계산함

        Note:
            sim(Q->D): Avg[Sum_{t in Q} (max t2 in D IC(MICA(t1, t2)))]
            (https://user-images.githubusercontent.com/86948539/225186395-bbc775c6-1df4-4185-b118-fd37e188fa19.png)

        Args:
            phenotypes1 (Set[Phenotype]): Phenotype 개념의 집합
            phenotypes2 (Set[Phenotype]): Phenotype 개념의 집합

        Raises:
            NotImplementedError: 서브클레싱하여 만든 클레스에서 해당 함수가 정의되지
                않은 경우
        """
        raise NotImplementedError

    @abstractmethod
    def get_semantic_similarity(
        self, phenotypes1: Set[HPO], phenotypes2: Set[HPO]
    ) -> float:
        """두 집합개념 (concept1, concept2)의 ontology에서 의미론적 유사도를 계산함.

        self.get_semantic_similarity_one_side을 호출하여, 양방향의 의미론적 유사도를
        계산함.

        Args:
            phenotypes1 (Set[Phenotype]): Phenotype 개념의 집합
            phenotypes2 (Set[Phenotype]): Phenotype 개념의 집합

        Raises:
            NotImplementedError: 서브클레싱하여 만든 클레스에서 해당 함수가 정의되지
                않은 경우
        """

        raise NotImplementedError

    def get_disease_similarty(self, patient_hpos: Set[HPO]) -> Dict[str, float]:
        """환자의 Phenotype의 집합 주어졌을 때의, Phenotype에 등재된 질환들 사이의 질병유사도를
        계산하여 반환

        Note:
            유사도 점수의 범위는 [0, inf).

        Args:
            patient_hpos (set): 환자의 Phenotype의 집합

        Returns:
            dict: 질환별 환자증상-질환증상의 유사도

        Example:
            >>> self.get_disease_similarity(patient_hpos={....})
            {
                "OMIM:619344": 0.58,
                "OMIM:619345": 0.48,
                "OMIM:619346": 5.38,
                "OMIM:619347": 0.28,
                "OMIM:619348": 0.18,
                "OMIM:619349": 0.08,
                "OMIM:619340": 1.99,
                "OMIM:619350": 0.88,
                "OMIM:619341": 0.78,
                "OMIM:619343": 0.68,
            }
        """
        disease_similarity: Dict[str, float] = dict()
        for disease_id, disease_obj in self.disease_pheno_map.items():
            disease_similarity[disease_id] = self.get_semantic_similarity(
                disease_obj.pheno, patient_hpos
            )

        return disease_similarity
