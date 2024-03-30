import glob

from .io_ops import read_json
from .data_model import Patient, Patients, Ontology

def load_phenopacket_patients(phenopacket_dir:str, ontology:Ontology) -> Patients:
    benchmark_patients_container = list()
    for path in glob.glob(f"{phenopacket_dir}/*"):
        p_data = read_json(path)
        
        patient_id = p_data["id"]
        disease_ids = {disease["term"]["id"] for disease in p_data["diseases"]}
        hpos = ontology[[phenotype["type"]["id"] for phenotype in p_data["phenotypicFeatures"]]]

        benchmark_patients_container.append(
            Patient(
                id=patient_id,
                hpos=hpos,
                disease_ids=disease_ids
            )
        )
        
    return Patients(benchmark_patients_container)

def get_sym(patient, disease, similarity_matrix):
    sum_sym = 0.
    sum_ic = 0.
    for p_sym in patient.hpos:
        max_sim = 0.
        for d_sym in disease.hpos:
            try:
                score = similarity_matrix[p_sym.id][d_sym.id] 
            except:
                score = 0 
                
            if score > max_sim:
                max_sim = score

        sum_sym += max_sim * p_sym.ic
        sum_ic += p_sym.ic
    
    return sum_sym, sum_ic


def get_pheno2disease(patient, disease):
    sum_sym_p, sum_ic_p = get_sym(patient, disease)
    sum_sym_d, sum_ic_d = get_sym(disease, patient)
    sym_pd = (sum_sym_p+sum_sym_d) / (sum_ic_p + sum_ic_d)

    pheno2disease = sym_pd + (sum_sym_p/sum_ic_p)

    return pheno2disease