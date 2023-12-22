from __future__ import annotations
from functools import reduce
from typing import Dict, List, Any, Set, Iterable, Union
from dataclasses import dataclass, field
from collections import defaultdict

import math
import pronto
import numpy as np
from ontology_src import SORUCE_URL

import sys

sys.setrecursionlimit(20000)


class Base:
    """증상유사도 데이터클레스의 베이스클레스

    Example:
        >>> class HPOs(Base):
        >>>     ...
        >>> hpos = HPOs([HPO(..)])
        >>> len(hpos)
        >>> for hpo in hpos:
        >>>     hpo.id
    """

    data: List[Any]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, query: Any) -> Any:
        return self.data[query]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self) -> HPO:
        """iteration을 위한 magic method

        Raises:
            StopIteration

        Returns:
            HPO

        Example:
            >>> hpos = HPOs(...)
            >>> for hpo in hpos:
                    hpo.name
                    ...

        """
        if self._index < len(self.data):
            result = self.data[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration


@dataclass
class HPO:
    """HPO (node)"""

    id: str
    name: str = str()
    definition: str = str()
    synonyms: Set[str] = field(default_factory=set)
    xref: Set[str] = field(default_factory=set)
    vector: np.ndarray = np.empty(shape=(1,))
    depth: int = 0
    ic: float = -1

    subclasses: Set[HPO] = field(default_factory=set, repr=False)
    ancestors: Set[HPO] = field(default_factory=set, repr=False)

    def __hash__(self) -> int:
        return hash(self.id)

    def _get_subclasses_recursively(self, subclass: HPO, container: set) -> Set[HPO]:
        for sub_subclass in subclass.subclasses:
            container.add(sub_subclass.id)
            self._get_subclasses_recursively(sub_subclass, container)

        return container

    @property
    def all_subclasses(self) -> List[HPO]:
        """현재 클래스를 포함하여 모든 하위 클래스를 찾아 반환
        Return
            List[HPO]: 모든 하위 클래스의 ID 목록
        Example:
            >>> self.all_subclasses
            [
                HPO(...),
                HPO(...),
                ...
                HPO(...),
            ]
        """
        if hasattr(self, "_all_subclasses"):
            return self._all_subclasses

        container = set()
        self._get_subclasses_recursively(self, container)

        self._all_subclasses = [HPO(hpo_id) for hpo_id in container]
        return self._all_subclasses

    def _get_ancestors_recursively(self, ancestor: HPO, container: set) -> Set[HPO]:
        for ancestors in ancestor.ancestors:
            container.add(ancestors.id)
            self._get_ancestors_recursively(ancestors, container)

        return container

    @property
    def all_ancestors(self) -> List[HPO]:
        """현재 클래스를 포함하여 모든 상위 클래스를 찾아 반환
        Return
            List[HPO]: 모든 상위 클래스의 ID 목록
        Example:
            >>> self.all_ancestors
            [
                HPO(...),
                HPO(...),
                ...
                HPO(...),
            ]
        """
        if hasattr(self, "_all_ancestors"):
            return self._all_ancestors

        container = set()
        self._get_ancestors_recursively(self, container)
        self._all_subclasses = [HPO(hpo_id) for hpo_id in container]
        return self._all_subclasses


@dataclass
class HPOs(Base):
    """복수의 HPO을 포함하고 있는 객체

    Example:
        >>> hpos = HPOs([hpo(...) for hpo in _phos])

        /// Membership
        >>> "HP:0000005" in hpos
        True
        >>> "Abnormality of hand" in hpos
        True

        /// Indexing
        >>> hpos["HP:0000005"]
        HPO(id="HP:0000005", ...)

        /// iteration
        >>> all_vectors = list()
        >>> for hpo in hpos:
        >>>     all_vectors.append(hpo.vector)
        >>> np.vstack(all_vectors)


    """

    data: List[HPO]

    def __post_init__(self):
        self.id2hpo = dict()
        self.name2hpo = dict()

        for hpo in self.data:
            self.name2hpo[hpo.name] = hpo
            self.id2hpo[hpo.id] = hpo

    def __getitem__(self, query: Any) -> HPO:
        if isinstance(query, str):
            if query.startswith("HP:"):
                if query not in self.id2hpo:
                    raise IndexError(f"Passed item ({query}) not found")

                return self.id2hpo[query]

            if query not in self.name2hpo:
                raise IndexError(f"Passed item ({query}) not found")

            return self.name2hpo[query]

        if isinstance(query, int):
            return self.data[query]

        if isinstance(query, Iterable):
            return HPOs([self[q] for q in query])

        return self.data[query]

    def __contains__(self, query) -> bool:
        """In을 위한 magic method

        Raises:
            StopIteration

        Returns:
            bool

        Example:
            >>> hpos = HPOs(...)
            >>> "HP:0000005" in hpos
            False

        """
        if query.startswith("HP:"):
            return query in self.id2hpo

        if isinstance(query, str):
            return query in self.name2hpo

        if isinstance(query, HPO):
            return query.id in self

    def __repr__(self) -> str:
        return f"HPOs(N HPO={len(self.data)})"

    @property
    def vector(self) -> np.ndarray:
        return np.vstack([hpo.vector for hpo in self.data])

    def sort_by_depth(self) -> HPOs:
        return sorted(self.data, key=lambda hpo: hpo.depth)

    @property
    def max_depth(self) -> int:
        if hasattr(self, "_max_depth"):
            return self._max_depth

        depthest_hpo: HPO = self.sort_by_depth()[-1]
        self._max_depth = depthest_hpo.depth
        return self._max_depth


@dataclass
class Ontology(HPOs):
    """온톨로지(Signleton pattern) 데이터클레스

    Note:
        실행순서
        1. __new__ : 메모리에 할당. 클레스변수로 인스턴스를 생성하여 저장
        2. __init__: 생성된 인스턴스를 초기화
        3. __post_init__: 초기화 후 후처리

    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton 패턴을 위한 온톨로지

        Example:
            >>> vectorized_hpos = load_pickle(ARTIFACT_PATH["hpo_name"])
            >>> my_ontology1 = Ontology(vectorized_hpos)
            >>> my_ontology2 = Ontology(vectorized_hpos)
            >>> my_ontology3 = Ontology(vectorized_hpos)

            >>> print(id(my_ontology))
            140383158424480
            >>> print(id(my_ontology2))
            140383158424480
            >>> print(id(my_ontology3))
            140383158424480
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
            return cls._instance

        return cls._instance

    def __init__(self, vectorized_hpos: HPOs):
        super().__init__(vectorized_hpos.data)
        ontology = pronto.Ontology(SORUCE_URL["hpo_obo"])
        self._update_ancestors(ontology)
        self._update_subclasses(ontology)
        self._update_depth(ontology)
        self._update_information_contents()

    def __repr__(self) -> str:
        return f"Ontology(N HPO={len(self.data)})"

    def __call__(self, hpo_ids: List[str]) -> HPOs:
        """HPO 객체를 생성하는 메서드

        Args:
            hpo_ids (List[str]): HPO ids

        Returns:
            HPOs

        Example:
           >>> ontology = Ontology(...)
           >>> patient_hpos = my_ontology(["HP:0000005", "HP:0000006"])
           >>> patient_hpos.vector
           array(
                [
                    [-0.00792581,  0.01476753, -0.00463055, ..., -0.00431325,
                    0.01031215, -0.02622988],
                    [-0.00972291,  0.02108144, -0.00561761, ..., -0.00913226,
                    0.00773679, -0.01496731]
                ]
           )
        """
        return HPOs([self[hpo_id] for hpo_id in hpo_ids])

    def _update_subclass_level(
        self,
        hpo: HPO,
        hpo_ontology: pronto.Ontology,
        phenotype2level: Dict[str, List[int]],
    ) -> None:
        """주어진 Phenotype에 대해서, 바로 아래의 자손의 Phenotype의 레벨을
        Phenotype2level에 업데이트함

        Args:
            hpo (HPO): 서브클레스
            phenotype (Phenotype): 표현형 개념
            phenotype2level (Dict[str, int]): phenotype-level 매핑딕셔너리

        """
        term: pronto.term.Term = hpo_ontology[hpo.id]
        level = max(phenotype2level[hpo.id])
        for subterm_id in term.subclasses(distance=1, with_self=False).to_set().ids:
            phenotype2level[subterm_id].append(level + 1)
            self._update_subclass_level(HPO(subterm_id), hpo_ontology, phenotype2level)

        return

    def _update_depth(self, hpo_ontology: pronto.Ontology) -> None:
        """HPOs에 저장된 HPO들의 depth을 업데이트함

        Args:
            hpo_ontology (pronto.Ontology): pronto에서 불러온 온톨로지

        Example:
            >>> import pronto
            >>> hpos = HPOs(...)
            >>> hpo_ontology = pronto.Ontology("http://..")
            >>> self.update_depth(hpo_ontology)
            >>> hpo = self["HP:0000005"]
            >>> print(hpo.depth)
            4
        """
        root_hpo_id = "HP:0000001"
        phenotype2level = defaultdict(list)
        phenotype2level[root_hpo_id].append(1)

        self._update_subclass_level(HPO(root_hpo_id), hpo_ontology, phenotype2level)

        for hpo_id, levels in phenotype2level.items():
            self[hpo_id].depth = max(levels)

        return

    def _update_subclasses(self, hpo_ontology: pronto.Ontology) -> None:
        """각 HPO의 서브클레스를 call-by-ref로 업데이트

        Args:
            hpo_ontology (_type_): pronto HPO ontology

        Example:
            >>> hpos = HPOs([...])
            >>> hpos.update_subclasses(hpo_ontology)
            >>> hpos["HP:0000001"].subclasses
        """
        for term in hpo_ontology.terms():
            self[term.id].subclasses = [
                self[hpo_id]
                for hpo_id in set(
                    term.subclasses(distance=1, with_self=False).to_set().ids
                )
            ]

        return

    def _update_ancestors(self, hpo_ontology: pronto.Ontology) -> None:
        """각 HPO의 서브클레스를 call-by-ref로 업데이트

        Args:
            hpo_ontology (_type_): pronto HPO ontology

        Example:
            >>> hpos = HPOs([...])
            >>> hpos.update_ancestors(hpo_ontology)
            >>> hpos["HP:0000001"].ancestors
        """
        for term in hpo_ontology.terms():
            self[term.id].ancestors = [
                self[hpo_id]
                for hpo_id in set(
                    term.superclasses(distance=1, with_self=False).to_set().ids
                )
            ]

        return

    def _update_information_contents(self) -> None:
        """개념(phenotype)의 IC값을 계산하여 업데이트

        Note:
            RelativeBestPair method
            # p(x) = |I(x)| / |I(T)|
                , where I(x) 하위 개념을 포함한 개수, |I(T)|: 전체 개념의 수
            # IC(x) = -log(p(x))
            reference: https://url.kr/bnyshe

        Args:
            phenotype (Phenotype): Phenotype ID

        """
        n_data = len(self.data)
        for hpo in self:
            n_subclass = len(hpo.all_subclasses)
            hpo.ic = -math.log(
                n_subclass / len(self.data) if n_subclass != 0 else 1 / n_data
            )

        return

    def _get_all_ancestor_ids(self, hpo, ancestor_ids: set) -> List[HPO]:
        ancestor_ids = set()
        for ancestor in hpo.ancestors:
            ancestor_ids.add(ancestor.id)
            self._get_all_ancestor_ids(ancestor, ancestor_ids)

        return ancestor_ids

    def get_most_informative_common_ancestor(
        self, hpo1: HPO, hpo2: HPO
    ) -> Union[HPO, None]:
        """두 개념의 공통조상 중 가장 정보력이 높은(MICA)의 개념을 불러옴

        Args:
            hpo1 (HPO): hpo내 개념에 해당하는 hpo ID
            hpo2 (HPO): hpo내 개념에 해당하는 hpo ID

        Returns:
            (hpo): MICA에 해당하는 hpo ID
        """

        hpo1_ancestor, hpo2_ancestor = set(), set()
        hpo1_ancestor = self._get_all_ancestor_ids(hpo1, hpo1_ancestor)
        hpo2_ancestor = self._get_all_ancestor_ids(hpo2, hpo2_ancestor)

        common_ancestor_id = hpo1_ancestor & hpo2_ancestor
        if common_ancestor_id is None:
            return None

        max_depth = 0
        mica = hpo1 if hpo1.depth >= hpo2.depth else hpo2

        for ancestor_id in common_ancestor_id:
            mica_candidate: HPO = self[ancestor_id]
            if mica_candidate.depth > max_depth:
                max_depth = mica_candidate.depth
                mica = mica_candidate

        return mica


@dataclass
class Patient:
    """환자 1명의 데이터클레스"""

    id: str
    hpos: HPOs
    disease_ids: Set[str]

    def __repr__(self) -> str:
        hpo_ids = ",".join([hpo.id for hpo in self.hpos])
        disease_id = ",".join(self.disease_ids)
        return f"Patient(id={self.id}, hpos={hpo_ids}, disease_id={disease_id})"


@dataclass
class Patients(Base):
    """복수의 Patients을 담은 데이터클레스"""

    data: List[Patient]

    def __post_init__(self):
        self.id2patient = {patient.id: patient for patient in self.data}

    def __getitem__(self, query: Any) -> Union[Patient, Patients]:
        if isinstance(query, str):
            if query not in self.id2patient:
                raise IndexError(f"Index ({query}) not in patients")

            return self.id2patient[query]

        if isinstance(query, int):
            return self.data[query]

        if isinstance(query, Iterable):
            return Patients([self[_query] for _query in query])

    def __repr__(self) -> str:
        return f"Patients(N={len(self.data)})"

    @property
    def all_confirmed_diseases(self) -> set:
        return reduce(set.union, [patient.disease_ids for patient in self])


@dataclass
class ClinicalSynopsis:
    """OMIM disease의 clinical synopsis"""

    names: Set[str]
    vector: np.ndarray


@dataclass
class Disease:
    """1개의 질환에 대한 데이터 클레스"""

    id: str
    name: str
    hpos: HPOs
    clinical_synopsis: ClinicalSynopsis = None


@dataclass
class Diseases(Base):
    data: List[Disease]

    def __post_init__(self):
        self.id2disease = {disease.id: disease for disease in self.data}
        self.all_symptom_vectors = np.concatenate([d.hpos.vector for d in self.data])
        self.n_symptoms = len(self.all_symptom_vectors)

    def __getitem__(self, query: Any) -> Union[Disease, Diseases]:
        if isinstance(query, str):
            if query not in self.id2disease:
                raise IndexError(f"Index ({query}) not in disease")

            return self.id2disease[query]

        if isinstance(query, int):
            return self.data[query]

        if isinstance(query, Iterable):
            return Diseases([self[_query] for _query in query])

    def __repr__(self) -> str:
        return f"Diseases(N={len(self.data)})"

    @property
    def all_disease_ids(self) -> set:
        return set(self.id2disease.keys())
