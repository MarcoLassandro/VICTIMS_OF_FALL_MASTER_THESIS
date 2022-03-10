import pandas as pd
from experiment_tools import *
import json

working_path = "C:\\Users\\NileNile\\Desktop\\REPOSITORY\\VICTIMS_OF_FALL_MASTER_THESIS"


experiment_name = "Experiment_10_08_12_21"
with open(f"{working_path}\\input_exp\\{experiment_name}.json", "r") as f:
    data=f.read().replace('\n', '')

experiment_setup = json.loads(data)

path_str = f"{working_path}\\data\\victims_of_fall_V2.csv"
df = pd.read_csv(path_str)
binarize = None if "binarize" not in experiment_setup.keys() else eval(experiment_setup["binarize"])

dataset_list = Experiment.get_dataset(df, experiment_setup["dataset_settings"])
print(experiment_setup["dataset_settings"])
print(dataset_list[0])