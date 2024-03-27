import time
from pathlib import Path
from logging import Logger
from typing import Dict, List, Set, Literal

import gzip
import numpy as np
import pandas as pd

from core.data_model import CNVData, Variant
from SemanticSimilarity.calculator import cal_symtpom_similarity_from_lambda


class CNVFeaturizer:
    cnv_root_path = Path("/NAS/data/Analysis")
    cnv_38_root_path = Path("/GDS/data/EVIDENCE.tucuxi/result_sv_grch38")
    hpo2disease_path = Path("/DAS/data/DB/processedData/disease/result/disease.txt.gz")
    cnv_new_path = Path("/DAS/data/EVIDENCE.tucuxi/result_cnv")
    header = ["ACMG_bayesian", "symptom_similarity", "num_genes"]

    def __init__(
        self,
        sequencing: Literal["wes", "wgs", "all"] = "wes",
        logger: Logger = Logger(__name__),
    ):
        self.logger = logger
        self.sequencing = sequencing
        self.hpo2disease_df = pd.read_csv(
            self.hpo2disease_path,
            compression="gzip",
            sep="\t",
            usecols=["#omimPhenoId", "geneSymbol"],
        )

    def get_symptom_similarity(
        self, gene_symbols: Set[str], disease_similiarty: Dict[str, float]
    ) -> float:
        # patient's cnv associated disease
        gene_symbol_matched = self.hpo2disease_df["geneSymbol"].apply(lambda x: x in gene_symbols)
        matched_df: pd.DataFrame = self.hpo2disease_df[gene_symbol_matched]
        matched_df = matched_df[matched_df["#omimPhenoId"] != "-"]
        if len(matched_df) == 0:
            return 0
        matched_df["#omimPhenoId"] = matched_df["#omimPhenoId"].apply(lambda x: f"OMIM:{x}")

        # query cnv disease in df
        matched_df["semantic_similarity"] = matched_df["#omimPhenoId"].apply(
            lambda omimid: disease_similiarty.get(omimid, 0)
        )
        # query 결과 내에 omimPhenoId 존재하지않는 경우 0으로 대치
        max_sim = matched_df["semantic_similarity"].astype(np.float32).max()
        del matched_df

        return max_sim

    def build_data_new(self, sample_id: str, causal_variant: List[str] = list()) -> CNVData:
        """CNV 데이터를 생성하여 반환

        Note:
            2023-05-09 이후부터는 CNV 데이터를 /DAS 이하에 저장하고, json포맷대신에
            tsv파일을 저장함. 그리고 이 tsv파일을 S3 버켓에 올림.

            --TSV파일 포맷--
            ##Called: Conifer=14;Manta=0;3bCNV=8;Total=22
            ##Remained: Conifer=2;Manta=0;3bCNV=1;Total=3
            #chrom  start   end     cnv_type        caller  quality genotype        allele_depth...
            15      23606367        28623955        DEL     Conifer -       -       -       -   ...
            15      23606367        28623955        DEL     Conifer -       -       -       -   ...
            15      23606367        28623955        DEL     Conifer -       -       -       -   ...
            15      23606367        28623955        DEL     Conifer -       -       -       -   ...
            15      23606367        28623955        DEL     Conifer -       -       -       -   ...

            --features--
            1. 유전자수: 변이가 커버하는 유전자수는 대표변이와 동일한 변이가 커버하는 유전자수를 의미
            2. 증상유사도: 변이의 유전자의 여러 질환중, 증상유사도가 가장 높은 값
            3. ACMG score: AMCG 스코어의 최대값(동일한 변이면 모두 같은 듯)

        Args:
            sample_id (str): 샘플 ID 문자열.
            causal_variant (List[str], 선택적): 인과 유전자 변이 리스트. 기본값은 ["-"].

        Return:
            CNVData: CNV 데이터 객체.

        Examples:
            >>> featurizer = CNVFeaturizer(...)
            >>> featurizer.build_data("EPB23-LMKB", ["5:96244664-96271878"])
            CNVData(n_variants=3, causal_variant=["5:96244664-96271878"])
        """
        self.logger.debug(
            f"Passed sample_id({sample_id}), causal_variant({','.join(causal_variant)})"
        )

        filepath = CNVFeaturizer.cnv_new_path / sample_id / f"{sample_id}.uploaded.tsv.gz"

        try:
            skiprows = 0
            with gzip.open(filepath, "rt") as fh:
                while True:
                    line = fh.readline()
                    if not line.startswith("#chrom"):
                        skiprows += 1
                        continue

                    break

            df = pd.read_csv(
                filepath,
                sep="\t",
                compression="gzip",
                skiprows=skiprows,
            )

        except:
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )

        if df.empty:
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )

        x = list()
        y = list()
        variants_names = list()
        varaints = df.apply(lambda x: f"{x['#chrom']}:{x['start']}-{x['end']}", axis=1)
        for variant in set(varaints):
            variant_rows = df.loc[varaints == variant]

            if len(variant_rows) == 0:
                continue

            n_genes = {
                len(gene)
                for genes in variant_rows["all_coding_symbols"].apply(lambda x: x.split("::"))
                for gene in genes
                if gene != list() and gene != ["-"]
            }
            similarities = [
                float(sim)
                for sim in variant_rows["symptom_similarity"]
                if sim != "-" and sim != "."
            ]
            rules = variant_rows["merged_acmg_rules"].tolist()[0]
            score = max(
                [float(score) for score in set(variant_rows["merged_acmg_sum"]) if score != "-"]
            )
            max_n_gene = max(n_genes)
            symptom_similarity = max(similarities) if similarities else 0

            x.append([score, symptom_similarity, max_n_gene])
            y.append(False if variant not in causal_variant else True)
            variants_names.append(Variant(variant, acmg_rules=rules))

        # 원인변이는 있는데, 데이터셋내에 존재하지 않는 경우는 제외
        if causal_variant and sum(y) == 0:
            self.logger.warning("Causal CNV of sample(%s) not found." % sample_id)
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )

        return CNVData(
            causal_variant=causal_variant,
            x=np.array(x).astype(np.float32),
            y=np.array(y).astype(np.float32),
            variants=variants_names,
            header=self.header,
        )

    def build_data(self, sample_id, causal_variant: List[str], patient_hpos: List[str]):
        """
        Build data for the given sample using the provided causal variants and patient phenotypes.

        Note:
            This method is responsible for building data for the specified sample, taking into account
            the causal variants and patient phenotypes to generate the appropriate features for the
            featurizer.

        Args:
            sample_id (str): The identifier of the sample for which the data is being built.
            causal_variant (List[str]): A list of strings representing the genomic positions of causal variants.
            patient_hpos (List[str]): A list of strings representing the Human Phenotype Ontology (HPO) terms associated with the patient.

        Returns:
            CNVData: An instance of the CNVData class containing the generated features for the sample.

        Examples:
            >>> featurizer = CNVFeaturizer(...)
            >>> featurizer.build_data("EPB23-LMKB", ["5:96244664-96271878"],
            ...                      ["HP:0000494", "HP:0000316", ...])
            CNVData(n_variants=3, causal_variant=["5:96244664-96271878"])
        """
        # TODO
        if self.sequencing in ["wgs", "all"]:
            raise NotImplementedError("wgs sample is currently not supported!")

        # 신규 CNV데이터셋 우선
        cnv_data = self.build_data_new(sample_id, causal_variant=causal_variant)

        dummy_array = -np.ones((1, len(self.header))).astype(np.float32)
        if not np.equal(cnv_data.x, dummy_array).all():
            return cnv_data

        # 초기 CNV데이터셋
        cnv_dir: Path = self.cnv_root_path / sample_id / "CNV"
        cnv_file_path: Path = cnv_dir / f"{sample_id}.result.acmg.json"

        if not (cnv_file_path.is_file()):
            cnv_file_path = cnv_dir / f"{sample_id}.3bcnv.result.acmg.json"

        try:
            patient_cnv_df = pd.read_json(cnv_file_path)
        except ValueError:
            self.logger.debug(f"{sample_id} is passed due to corrupted file.")
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )
        except FileNotFoundError:
            self.logger.debug(f"{sample_id} is passed due to NotFoundFile.")
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )

        patient_cnv_df["label"] = False
        if "pos" not in patient_cnv_df:
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )
        patient_cnv_df.loc[patient_cnv_df["pos"].isin(causal_variant), "label"] = True

        disease_similiarty = dict()
        trials = 0
        while trials < 5:
            try:
                disease_similiarty: dict = cal_symtpom_similarity_from_lambda(patient_hpos)
            except:
                time.sleep(5)

            trials += 1

        if not disease_similiarty:
            self.logger.warning(f"Failed to calculate symptom similarity for {sample_id}")
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )

        patient_cnv_df["symptom_similarity"] = patient_cnv_df["genes"].apply(
            lambda x: self.get_symptom_similarity(x, disease_similiarty)
        )

        return CNVData(
            causal_variant=causal_variant,
            x=patient_cnv_df[["score", "symptom_similarity", "num_genes"]].values.astype(
                np.float32
            ),
            y=patient_cnv_df["label"].values.astype(np.float32),
            header=self.header,
        )
