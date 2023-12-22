"""환자 데이터셋 빌드"""
import os
import sys
from typing import Set

import tqdm

from core.data_model import HPOs, Ontology, Patient, Patients
from core_3asc.dynamodb_ops import DynamoDBClient
from core_3asc.data_model import Report
from core.io_ops import load_pickle, save_pickle
from log_ops import get_logger
from ontology_src import ARTIFACT_PATH

sys.setrecursionlimit(20000)  # HPO개수만큼 리커젼필요
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")

if __name__ == "__main__":
    logger = get_logger("data_build")
    dynamodb_client = DynamoDBClient(os.path.join(DATA_DIR, "keyfile.yaml"))

    found_sample_ids: Set[str] = dynamodb_client.get_found_samples(
        include_inconclusive=True
    )
    logger.info("Total N=%s found samples collected" % str(len(found_sample_ids)))

    logger.info("Start building ontology")
    vectorized_hpo: HPOs = load_pickle(ARTIFACT_PATH["hpo_definition"])
    ontology = Ontology(vectorized_hpo)
    logger.info("End of building ontology")

    patient_container = list()
    for sample_id in tqdm.tqdm(list(found_sample_ids)):
        logger.debug("sample_id(%s) collecting" % sample_id)
        report: Report = dynamodb_client.get_report(sample_id)
        if not report.snv_variant:
            logger.debug("sample_id(%s) snv_variant not found" % sample_id)
            continue

        hpos: Set[str] = dynamodb_client.get_hpo(sample_id)
        vectorized_hpo: HPOs = ontology([hpo for hpo in hpos])
        disease_ids = {variant.disease_id for variant in report.snv_variant}

        patient = Patient(sample_id, disease_ids=disease_ids, hpos=vectorized_hpo)
        patient_container.append(patient)

    logger.info("Total N=(%s) samples collected" % str(len(patient_container)))
    patients = Patients(patient_container)

    logger.info("Save: %s" % ARTIFACT_PATH["patients"])
    save_pickle(patients, ARTIFACT_PATH["patients"])
