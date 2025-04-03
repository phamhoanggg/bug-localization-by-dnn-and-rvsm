from dnn_model import dnn_model_kfold
from rvsm_model import rsvm_model
from datasets import DATASET
print(dnn_model_kfold(DATASET.features, DATASET.bug_repo, 10))
print(rsvm_model(DATASET.features, DATASET.bug_repo))