import mlflow

TRACKING_URI = "http://182.208.81.130:16003"
EXP_SYMPTOM = "SemanticSimilarity"

def get_experiment():
    mlflow.set_tracking_uri(TRACKING_URI)
    
    client = mlflow.tracking.MlflowClient(TRACKING_URI)
    experiment = client.get_experiment_by_name(EXP_SYMPTOM)
    
    if not experiment:
        client.create_experiment(EXP_SYMPTOM)
        return client.get_experiment_by_name(EXP_SYMPTOM)
    
    return experiment
    