# Semantic similarity between patient's HPOs and rare diseases' HPOs


#### Embedding vector 구하기
1. HPO definition에 대한 embedding vector 구하기
```/bin/bash
$ python3 concept_embedding.py -k [OpenAI key] -e hpo_definition -i 0.18 --resume
```

2. HPO name 대한 embedding vector 구하기
```/bin/bash
$ python3 concept_embedding.py -k [OpenAI key] -e hpo_name -i 0.18 --resume
```




## HPO 핸들링
#### 1. 온톨로지 생성
```python3
>>> from core.data_model import Ontology
>>> from core.io_ops import load_pickle
>>> from ontology_src import ARTIFACT_PATH

>>> hpos = load_pickle(ARTIFACT_PATH["hpo_name"])
>>> my_ontology = Ontology(hpos)
```

#### 2. HPO의 집합 생성(HPOs) 및 벡터화
```python3
>>> patient_hpos = my_ontology(["HP:0000005", "HP:0000006"])
>>> patient_hpos.vector
array([[-0.00792581,  0.01476753, -0.00463055, ..., -0.00431325,
         0.01031215, -0.02622988],
       [-0.00972291,  0.02108144, -0.00561761, ..., -0.00913226,
         0.00773679, -0.01496731]])
```

#### 3. Iteration
```python3
>>> for hpo in hpos:
>>>     print(hpo)
HPO(id='HP:0000005', name='Mode of inheritance', ...)
HPO(id='HP:0000006', name='Autosomal dominant inheritance', ...,)
```

#### 4. Membership
```python3
>>> "HP:0000005" in patient_hpos
True
>>> HPO(id="HP:0000005",...) in patient_hpos
True
```

#### 5. Index(getitem)
```python3
>>> patient_hpos["HP:0000005"]
HPO(id='HP:0000005', name='Mode of inheritance', ...)
```