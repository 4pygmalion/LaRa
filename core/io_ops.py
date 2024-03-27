import pickle
from typing import Any
import json


def save_pickle(object, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(object, fh)

    return


def load_pickle(path: str) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)

def read_json(path:str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)