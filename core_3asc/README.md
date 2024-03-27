### 3Billion AI data model (SNV, CNV)

<br>

### Contents
```
├── cnv_factory.py : CNV 특징값을 추출하는 모듈 (증상유사도를 불러와야해서, cytosine)
├── snv_factory.py : SNV 특징 값을 추출하는 모듈
├── databuild.py : snv_factory.py, cnv_factor.py을 이용하여 bag, instance label 및 데이터를 직렬화
├── data_model.py : CNV, SNV, Patient등에 대한 데이터 모델
├── network.py : MIL network
└── README.md
```