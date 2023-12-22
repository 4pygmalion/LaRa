"""Concept들의 임베딩벡터를 구함

Note:
    - OpenAI Request Limit이 Tier에 따라 분당, 일당 요청수가 다름

Example
    // HPO 이름에 대한 임베딩 구하기
    $ python3 concept_embedding.py \
        -k [openai_key] \
        -e hpo_name \
        -i 0.3
        
    //
"""

import os
import time
import json
import argparse
import requests
from requests.exceptions import RequestException
from datetime import datetime
import multiprocessing
from functools import partial

import tqdm
import numpy as np
import pronto

from log_ops import get_logger
from core.data_model import HPO, HPOs
from core.io_ops import load_pickle, save_pickle
from ontology_src import SORUCE_URL, ARTIFACT_PATH

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--openai_key", help="OpenAI key", type=str, required=True
    )
    parser.add_argument(
        "-e",
        "--embedding",
        choices=["hpo_name", "hpo_definition"],
        type=str,
        required=True,
    )
    parser.add_argument("-i", "--interval", type=float, default=0.2)
    parser.add_argument("-r", "--resume", action="store_true")
    return parser.parse_args()


def get_embedding(text: str, key: str, max_retries=10) -> np.ndarray:
    """
    Get a text embedding using the OpenAI GPT-3 API with retry logic.

    Parameters:
        text (str): The input text for which the embedding is requested.
        key (str): OpenAI API key for authentication.
        max_retries (int): Maximum number of retries in case of errors (default is 3).

    Returns:
        embedding_vector (np.ndarray): A NumPy array representing the text embedding.

    Raises:
        Exception: If there is an error during the API request after max_retries attempts.

    Example:
        >>> text = "This is an example text."
        >>> api_key = "your_openai_api_key"
        >>> embedding = get_embedding(text, api_key)
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    data = {"input": text, "model": "text-embedding-ada-002"}
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(
                url="https://api.openai.com/v1/embeddings",
                headers=headers,
                data=json.dumps(data),
            )

            # Check for success (2xx status code) and return the result
            if 200 <= response.status_code < 300:
                return np.array(response.json()["data"][0]["embedding"])

            # If the error code is 429, do not retry
            elif response.status_code == 429:
                raise Exception(f"Rate limit exceeded for text: {text}")

            # For other error codes, retry
            else:
                print(
                    f"Retrying request for text ({text}) - Status Code: {response.status_code}"
                )
                retries += 1
                time.sleep(1)

        except RequestException as e:
            retries += 1
            time.sleep(1)
            print(f"Requested text ({text}), Raise {e}")

    # If max_retries is reached and still not successful, raise an exception
    raise Exception(
        f"Failed to get embedding after {max_retries} retries for text: {text}"
    )


def update_embedding_vector(concept, property, key, interval, shared_vectors):
    time.sleep(interval)
    shared_vectors[concept.id] = get_embedding(getattr(concept, property), key)


def update_embedding_vectors(
    concepts: HPOs, property: str, key: str, interval: float = 0.05
):
    manager = multiprocessing.Manager()
    shared_vectors = manager.dict()

    update_partial = partial(
        update_embedding_vector,
        property=property,
        key=key,
        interval=interval,
        shared_vectors=shared_vectors,
    )
    with multiprocessing.Pool(20) as pool:
        try:
            list(
                tqdm.tqdm(
                    pool.imap_unordered(update_partial, concepts),
                    total=len(concepts),
                )
            )

        except:
            print("Exception raised. Terminating processing.")
            pool.terminate()
            pool.join()
            return

    # Retrieve the updated vectors from the shared dictionary
    for concept in concepts:
        concept.vector = shared_vectors.get(concept.id, np.zeros((1,)))

    return


def embedding_hpo_warpper(
    embedding: str, resume: bool, key: str, interval: float
) -> None:
    """HPO data에 대해서 임베딩

    Parameters:
        embedding (str): 임베딩할 속성 (e.g hpo_name, hpo_description)
        resume (bool): 이전 임베딩을 계속 할 경우(예, 1일 요청 10,000회 제한)
        key (str): OpenAI key
        interval (float): 요청과 요청사이의 interval


    Examples:
        >>> embedding_hpo_warpper("hpo_name", True, "example_key", 0.1)
    """
    output_path = ARTIFACT_PATH[embedding]
    concept, property = embedding.split("_")

    if resume:
        hpos: HPOs = load_pickle(output_path)
    else:
        ontology = pronto.Ontology(SORUCE_URL["hpo_obo"])
        hpos = HPOs(
            [
                HPO(
                    term.id,
                    name=term.name,
                    definition=str(term.definition),  # Pronto.Definition -> str
                    synonyms={synonyms.description for synonyms in term.synonyms},
                    xref={xref.id for xref in term.xrefs},
                )
                for term in ontology.terms()
            ]
        )

    update_embedding_vectors(
        hpos,
        property=property,
        key=key,
        interval=interval,
    )
    save_pickle(hpos, output_path)

    return


if __name__ == "__main__":
    ARGS = get_args()
    LOGGER = get_logger(
        "concept_embedding.py",
        log_path=os.path.join(
            ROOT_DIR,
            "logs",
            "concept_embedding.%s.log" % datetime.now().strftime("%Y-%m-%d-%H-%M"),
        ),
    )

    for arg_name, value in vars(ARGS).items():
        if arg_name == "openai_key":
            continue
        LOGGER.info("ARGS (%s): (%s)" % (arg_name, value))

    if ARGS.embedding.startswith("hpo"):
        embedding_hpo_warpper(
            ARGS.embedding,
            ARGS.resume,
            key=ARGS.openai_key,
            interval=ARGS.interval,
        )
        LOGGER.info("Saved file at: %s" % ARTIFACT_PATH[ARGS.embedding])

    ## TODO
    if ARGS.embedding.startswith("disease"):
        raise NotImplementedError
