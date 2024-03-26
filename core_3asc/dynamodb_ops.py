import os
import re
import sys
from datetime import datetime
from logging import Logger
from typing import Set, List, Dict, Union

import yaml
import boto3
from boto3.dynamodb.conditions import Key

from .data_model import Report, Variant


class DynamoDBClient:
    onset_mapper = {
        # Adult
        "Adult": "Adult",
        "Adults": "Adult",
        # Elderly
        "Elderly": "Elderly",
        "Eldery": "Elderly",
        # Unknown
        "Uncertain": "Unknown",
        "Unknown": "Unknown",
        # Adolescent, Antenatal, Childhood, Infancy, Neonatal
        "Adolescent": "Adolescent",
        "Antenatal": "Antenatal",
        "Childhood": "Childhood",
        "Infancy": "Infancy",
        "Neonatal": "Neonatal",
    }

    def __init__(self, kyefile_path: str, logger: Logger = Logger(__name__)) -> None:
        self.kyefile_path = kyefile_path
        self.logger = logger
        self._set_dynamodb_client()
        self.id_pattern = re.compile(r"[E|G][A-Z]{2}\d{2}\-[A-Z]{4}")

    def _set_dynamodb_client(self) -> None:
        """keyfile로부터 제공된 자격 증명을 기반으로 DynamoDB 리소스 클라이언트를 생성하여 반환합니다.

        매개변수:
            keyfile_path (str): DynamoDB 자격 증명이 포함된 YAML 파일의 경로.

        Note:
            boto3.resources.base.ServiceResource: DynamoDB 리소스 클라이언트.
        """
        with open(self.kyefile_path, "r") as fh:
            credentials = yaml.load(fh, Loader=yaml.FullLoader)

        self.dynamodb_client = boto3.resource(
            "dynamodb",
            region_name=credentials["DYNAMODB"]["REGION"],
            aws_access_key_id=credentials["DYNAMODB"]["ACCESS_KEY"],
            aws_secret_access_key=credentials["DYNAMODB"]["SECRET_KEY"],
        )

        return

    def get_all_sample_ids(self) -> Set[str]:
        """DynamoDB 클라이언트를 사용하여 "orderInformation" 테이블에서 모든 샘플 ID를 가져옵니다.

        Args:
            dynamodb_client (boto3.resources.base.ServiceResource): DynamoDB 리소스 클라이언트.

        Returns:
            Set[str]: "orderInformation" 테이블의 모든 샘플 ID를 담고 있는 집합(set).
        """
        order_management_table = self.dynamodb_client.Table("OrderManagement")
        response: Dict[str, List] = order_management_table.scan()

        all_sample_ids = set()
        for res in response["Items"]:
            if "id" not in res or not self.id_pattern.match(res["id"]):
                continue

            all_sample_ids.add(res["id"])

        while "LastEvaluatedKey" in response:
            response: dict = order_management_table.scan(
                ExclusiveStartKey=response["LastEvaluatedKey"]
            )
            for res in response["Items"]:
                if "id" not in res or not self.id_pattern.match(res["id"]):
                    continue

                all_sample_ids.add(res["id"])

        return all_sample_ids

    def get_found_samples(self, include_inconclusive: bool = True) -> Set[str]:
        """Tucuxi-Finding 테이블에서 검색 결과를 통해 발견된 샘플 ID들의 집합을 반환

        Args:
            include_inconclusive (bool, optional): inconclusive conclusion 포함여부

        Returns:
            Set[str]: found sample ids의 집합

        Example:
            >>> self.get_found_samples()
            {
                "EPA23-ATER",
                "EPA22-ATER",
                ...
            }
        """
        tucuxi_finding_table = self.dynamodb_client.Table("Tucuxi-Finding")
        response = tucuxi_finding_table.scan()
        data = response["Items"]
        while "LastEvaluatedKey" in response:
            response = tucuxi_finding_table.scan(
                ExclusiveStartKey=response["LastEvaluatedKey"]
            )
            data.extend(response["Items"])

        found_sample_ids = set()
        for record in data:
            if "conclusion" not in record:
                continue

            sample_id = record["SK"]
            conclusion = record["conclusion"]

            if (not include_inconclusive) and conclusion.lower() == "inconclusive":
                continue

            found_sample_ids.add(sample_id)

        return found_sample_ids

    def get_report(self, sample_id: str) -> Report:
        """주어진 sample_id를 사용하여 DynamoDB 데이터베이스에서 보고서 정보를 가져옴

        Note:
            conclusion의 key가 매번 다름
            {
                "PK": "REPORT#EPH21-GMPZ",
                "analysisTool": "gebra",
                "conclusion": "Positive",
                "createdAt": "2021-10-20T15:49:45.000Z"
                ...
            }

            또는
            {
                "lastApprovedEvidenceLog": {
                    "build": "37",
                    "createdAt": "2023-08-11T02:15:27.775Z",
                    "diagnosis": "inconclusive"
                    ...
                }
            }

            Dynamodb에 원인변이는 Tucuxi-Finding 테이블에 존재함
            {
                "PK": "19-5719819-T-C",
                "SK": "EPJ21-TYYB",
                "ad": "60,74",
                "alt": "C",
                "chr": "19",
                "conclusion": "Inconclusive",
                "consequence": "missense_variant",
                "createdAt": "2021-12-02T17:46:29.000Z",
                "customerID": "AmWs",
                "dp": "134",
                "ethnicity": "African/African-American",
                "gender": "Female",
                "geneId": "605490",
                "geneSymbol": "LONP1",
                ...
            }

        Args:
            sample_id (str): 보고서를 조회할 샘플의 3B ID

        Returns:
            Report: 가져온 보고서 정보로 생성된 Report 객체

        Example:
            >>> self.get_report("EPA23-ABCD")
            Report(
                sample_id="EPA23-ABCD",
                conclusion="positive",
                snv_variant=Variant(
                    cpra="2-327832-A-T",
                    disease_id="OMIM:123423",
                    acmg_rules=["PM2"]
                )
            )
        """
        self.logger.debug("Querying report of sample_id(%s)" % sample_id)

        table = self.dynamodb_client.Table("OrderManagement")
        res = table.get_item(Key={"PK": f"REPORT#{sample_id}"})

        if "Item" not in res:
            return Report(sample_id=sample_id)

        report_info = res["Item"]
        if "conclusion" not in report_info:
            conclusion = (
                report_info.get("lastApprovedEvidenceLog", dict())
                .get("diagnosis", "")
                .lower()
            )
        else:
            conclusion = report_info["conclusion"].lower()
        conclusion = conclusion.strip()

        report_date = (
            datetime.strptime(report_info["createdAt"][:10], "%Y-%m-%d")
            if "createdAt" in report_info
            else datetime.min
        )
        report_date = datetime(report_date.year, report_date.month, report_date.day)

        snv_variants = list()
        cnv_variants = list()

        finding_table = self.dynamodb_client.Table("Tucuxi-Finding")
        variants = finding_table.query(
            IndexName="sampleGSI", KeyConditionExpression=Key("SK").eq(sample_id)
        )
        if "Items" not in variants:
            return Report(sample_id=sample_id, conclusion=conclusion)

        for variant_info in variants["Items"]:
            if "alt" in variant_info:
                if "phenoId" not in variant_info:
                    continue

                disease_id = (
                    "OMIM:" + variant_info["phenoId"]
                    if variant_info["phenoId"].isdigit()
                    else variant_info["phenoId"]
                )

                snv_variants.append(
                    Variant(
                        cpra=variant_info["PK"],
                        disease_id=disease_id,
                        symbol=variant_info["geneSymbol"],
                    )
                )

            else:
                cnv_variants.append(
                    Variant(
                        cpra=variant_info["PK"],
                    )
                )

        return Report(
            sample_id=sample_id,
            conclusion=conclusion,
            snv_variant=snv_variants,
            cnv_variant=cnv_variants,
            report_date=report_date,
        )

    def get_hpo(self, sample_id: str) -> Set[str]:
        """환자(sample_id)의 HPO정보를 반환함.

        Args:
            sample_id (str): 샘플ID

        Returns:
            Set[str]: 환자 HPO의 집합
        """
        item = self.dynamodb_client.Table("OrderManagement").get_item(
            Key={"PK": f"ORDER#{sample_id}"}
        )
        if "Item" not in item or "symptoms" not in item["Item"]:
            return set()

        return {
            symptom["hpo"] for symptom in item["Item"]["symptoms"] if "hpo" in symptom
        }

    def get_clinical_info(self, sample_id: str) -> Dict[str, Union[List[Dict], str]]:
        order_table = self.dynamodb_client.Table("OrderManagement")

        try:
            # [{"title":, "hpo":, "onsetAge":}]
            query_result = order_table.get_item(Key={"PK": f"ORDER#{sample_id}"})[
                "Item"
            ]["symptoms"]
        except:
            msg = f"order info for {sample_id} is missing."
            self.logger.error(msg)
            raise ValueError(msg)

        patient_symptoms = [
            (
                x["hpo"],
                x.get("title", ""),
                self.onset_mapper.get(x.get("onsetAge", ""), ""),
            )
            for x in query_result
            if x.get("hpo") is not None
        ]

        if len(patient_symptoms) == 0:
            msg = f"hpo ids for patient symptoms are missing. (here's queried symptoms {query_result})"
            self.logger.error(msg)
            raise ValueError(msg)

        try:
            # "Item" "gender"
            gender_and_age = order_table.get_item(Key={"PK": f"SUBJECT#{sample_id}"})[
                "Item"
            ]
            gender = gender_and_age["gender"]

        except:
            gender = ""
            self.logger.warning(f"gender info for '{sample_id}' is missing.")

        return {"symptoms": patient_symptoms, "gender": gender}
