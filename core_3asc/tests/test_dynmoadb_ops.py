import os
import sys
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(TEST_DIR)
ROOT_DIR = os.path.dirname(UTILS_DIR)
sys.path.append(ROOT_DIR)

from core.data_model import SNVData, Variant
from core.dynamodb_ops import DynamoDBClient


@pytest.fixture(scope="module")
def client():
    with patch("core.dynamodb_ops.DynamoDBClient._set_dynamodb_client"):
        client = DynamoDBClient("keypath")
        client.dynamodb_client = Mock()

        return client


def test_get_report(client):
    client.dynamodb_client.Table().get_item.return_value = {
        "Item": {"conclusion": "positive", "createdAt": "2023-08-13T03:24:03.199Z"}
    }

    client.dynamodb_client.Table().query.return_value = {
        "Items": [
            {"PK": "1-100-T-A", "alt": "A", "phenoId": "203232", "geneSymbol": "ABCD"}
        ]
    }

    report = client.get_report("ETA23-ABCD")
    assert report.sample_id == "ETA23-ABCD"
    assert report.conclusion == "positive"
    assert report.snv_variant == [
        Variant(cpra="1-100-T-A", symbol="ABCD", disease_id="OMIM:203232")
    ]
    assert report.cnv_variant == list()
    assert report.report_date == datetime(2023, 8, 13)
