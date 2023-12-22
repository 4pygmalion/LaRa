"""질병 데이터셋 빌드"""
import os
import sys
from typing import Set

import tqdm
import pandas as pd

from core.data_model import HPO, HPOs, Ontology, Disease, Diseases
from core_3asc.dynamodb_ops import DynamoDBClient
from core_3asc.data_model import Report
from core.io_ops import load_pickle, save_pickle
from log_ops import get_logger
from ontology_src import ARTIFACT_PATH, SORUCE_URL

sys.setrecursionlimit(20000)  # HPO개수만큼 리커젼필요
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")

if __name__ == "__main__":
    logger = get_logger("data_build(Disease)")

    logger.info(f"load disease_to_hpo from {SORUCE_URL['disease_hpo']}")
    disease_to_hpo = pd.read_csv(SORUCE_URL["disease_hpo"], skiprows=4, sep="\t")
    disease_to_hpo = (
        disease_to_hpo.groupby(["database_id", "disease_name"])["hpo_id"]
        .agg(list)
        .reset_index()
    )

    logger.info("Total N=%s diseases collected" % str(len(disease_to_hpo)))

    logger.info("Start building ontology")
    vectorized_hpo: HPOs = load_pickle(ARTIFACT_PATH["hpo_definition"])
    ontology = Ontology(vectorized_hpo)
    logger.info("End of building ontology")

    disease_container = list()
    for i, (id_, name, hpos) in tqdm.tqdm(disease_to_hpo.iterrows()):
        vectorized_hpo: HPOs = ontology(hpos)
        disease = Disease(id_, name=name, hpos=vectorized_hpo)
        disease_container.append(disease)

    logger.info("Total N=(%s) diseases collected" % str(len(disease_container)))
    diseases = Diseases(disease_container)

    logger.info("Save: %s" % ARTIFACT_PATH["diseases"])
    save_pickle(diseases, ARTIFACT_PATH["diseases"])
