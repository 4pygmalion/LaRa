"""OMIM Disease에 임베딩 추가
$ python3 preprocessing/omim_embedding.py \
    -i data/diseases.pickle
    -k [OpenAI key]
"""
import os
import sys
import json
import argparse
import concurrent.futures
from typing import Set

sys.setrecursionlimit(20000)

import tqdm
from requests.exceptions import RequestException

PREP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PREP_DIR)
sys.path.append(ROOT_DIR)
from core.data_model import Diseases, ClinicalSynopsis, Disease
from core.io_ops import load_pickle, save_pickle
from concept_embedding import get_embedding
from ontology_src import SourceData, ARTIFACT_PATH
from log_ops import get_logger


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Serialized disease object path"
    )
    parser.add_argument("-k", "--key", type=str, required=True, help="OpenAI key")
    parser.add_argument("--n_workers", type=int, default=20)
    return parser.parse_args()


def extract_clinical_synopsis(path: str) -> Set[str]:
    """JSON에서 clinical synopsis을 파싱하여 얻음

    Note:
    {
    "omim": {
        "entryList": [
            {
                "entry": {
                    "clinicalSynopsis": {
                        "abdomenBiliaryTractExists": false,
                        "abdomenExists": true,
                        "abdomenExternalFeaturesExists": false,
                        "abdomenGastrointestinal": "Poor feeding {SNOMEDCT:78164000,299698007}",
                        "abdomenGastrointestinalExists": true,
                        "abdomenLiverExists": false,
                        "abdomenPancreasExists": false,
                        "abdomenSpleenExists": false,
                        "cardiovascularExists": true,
                        "cardiovascularHeart": "cardiovascularHeart": "Heart defects (40%) {SNOMEDCT:13213009};\n
                            Atrial septal defect {UMLS C0018817 HP:0001631} {HPO HP:0001631};\n
                            Ventricular septal defect  {HPO HP:0001629};\n
                            Pulmonary valve stenosis{UMLS C0034089 HP:0001642} {HPO HP:0001642};\n
                            Bicuspid aortic valve{HPO HP:0001647};\n
                            Aortic dilatation (reported in 1 patient) {HPO HP:0004942}",
                ...

    Args:
        path (str): omim json path

    Returns:
        Set[str]: set of clinical synopsis
    """
    with open(path, "r") as fh:
        data = json.load(fh)

    for entry_info in data["omim"]["entryList"]:
        if "entry" in entry_info and "clinicalSynopsis" in entry_info["entry"]:
            break

    synopsis_info = entry_info.get("entry", dict()).get("clinicalSynopsis", dict())
    if not synopsis_info:
        return set()

    res = set()
    for header, values in synopsis_info.items():
        header = header.lower()
        if (
            header.endswith("exists")
            or header.startswith("contributor")
            or header.startswith("creation")
            or header.startswith("edit")
            or header.startswith("epoch")
        ):
            continue

        try:
            if header == "oldformat":
                for name, value in values.items():
                    sanitized_value = value.split("{")[0].strip()
                    res.add(sanitized_value)

            else:
                for value in values.split("\n"):
                    sanitized_value = value.split("{")[0].strip()
                    res.add(sanitized_value)

        except:
            continue

    return res


def do_main_job(disease: Disease) -> Disease:
    if not disease.id.startswith("OMIM"):
        return

    disease_json_path = os.path.join(
        OMIM_JSON_DIR, f"{disease.id.lstrip('OMIM:')}.json"
    )

    if not os.path.exists(disease_json_path):
        return

    try:
        synopsis: Set[str] = extract_clinical_synopsis(disease_json_path)
    except:
        return

    try:
        vector = get_embedding(",".join(synopsis), key=ARGS.key)
    except RequestException:
        return

    clinical_synopsis = ClinicalSynopsis(names=synopsis, vector=vector)
    setattr(disease, "clinical_synopsis", clinical_synopsis)

    return disease


if __name__ == "__main__":
    ARGS = get_args()
    LOGGER = get_logger("omim_embedding")
    diseases: Diseases = load_pickle(ARGS.input)

    OMIM_JSON_DIR = SourceData.omim_json_dir

    diseases_with_synopsis = list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=ARGS.n_workers) as executor:
        for disease_with_synopsis in tqdm.tqdm(
            executor.map(do_main_job, diseases), total=len(diseases)
        ):
            diseases_with_synopsis.append(disease_with_synopsis)

    # update
    for disease_with_synopsis in diseases_with_synopsis:
        if disease_with_synopsis is None:
            continue

        diseases[
            disease_with_synopsis.id
        ].clinical_synopsis = disease_with_synopsis.clinical_synopsis

    save_pickle(diseases, ARTIFACT_PATH["diseases"])
    LOGGER.info("Saved: %s", ARTIFACT_PATH["diseases"])
