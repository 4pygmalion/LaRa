import os
from dataclasses import dataclass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

RELEASE_URL = (
    "https://github.com/obophenotype/human-phenotype-ontology/releases/download"
)
HPO_VERSION = "v2023-10-09"

SORUCE_URL = {
    "hpo_obo": f"{RELEASE_URL}/{HPO_VERSION}/hp.obo",
    "hpo_full": f"{RELEASE_URL}/{HPO_VERSION}/hp-full.json",
    "disease_hpo": f"{RELEASE_URL}/{HPO_VERSION}/phenotype.hpoa",
    "gene_to_hpo": f"{RELEASE_URL}/{HPO_VERSION}/genes_to_phenotype.txt",
}

# TODO: 데이터클레스로 변경하여 속성으로 관리(For Auto-complete)
ARTIFACT_PATH = {
    "hpo_name": os.path.join(DATA_DIR, "hpo_name.vector.pickle"),
    "hpo_definition": os.path.join(DATA_DIR, "hpo_definition.vector.pickle"),
    "disease_name": os.path.join(DATA_DIR, "disease_name.vector.pickle"),
    "disease_desc": os.path.join(DATA_DIR, "disease_desc.vector.pickle"),
    "patients": os.path.join(DATA_DIR, "patients.pickle"),
    "diseases": os.path.join(DATA_DIR, "diseases.pickle"),
}


@dataclass
class SourceData:
    omim_json_dir: str = "/DAS/data/DB/rawData/OMIM/result"
