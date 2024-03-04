import numpy as np
import pandas as pd

from .ChainedEstimator import ChainedEstimator
from .FuzzyCMeans import FuzzyCMeans
from .GranularBinaryClassifier import GranularBinaryClassifier
from .GroupedPCA import GroupedPCA
from .GroupedSVD import GroupedSVD
#from .KerasSklearnWrapper import KerasSklearnWrapper
from .MaskedPCA import MaskedPCA
from .MaskedSVD import MaskedSVD

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans, spectral_clustering, Birch
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso, RidgeClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR, SVC
from scikeras.wrappers import KerasClassifier, KerasRegressor

from cache_decorator import Cache
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, RandomizedSearchCV, StratifiedShuffleSplit
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from multiprocessing import cpu_count
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier
from mord import OrdinalRidge, LogisticAT, LogisticIT

from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError, Accuracy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers


cpu_1 = 14
cpu_2 = 14

def create_regression_model(optimizer='adam', dim=1, hidden_layer_sizes=(64,), num_layers=1):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(Dense(hidden_layer_sizes[i], input_dim=dim, activation='relu'))
        else:
            model.add(Dense(hidden_layer_sizes[i], activation='relu'))
    model.add(Dense(1, activation='linear'))  # Use 'linear' activation for regression
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def eval_params(params, X, mask):
    parsed_params = []
    for spaces in params:
        parsed_spaces = {}
        for k, v in spaces.items():
            parsed_spaces[k] = eval(v)
        parsed_params.append(parsed_spaces)
    return parsed_params

def get_dataset(df, dataset_settings):
    X = df.dropna()
    X.Sesso = pd.Categorical(X.Sesso)
    X.Sesso = X.Sesso.cat.codes

    no_injury_parts = {
        "TESTA": 5,
        "TORACE": 5,
        "ADDOME": 5,
        "SCHELETRO": 5
    }

    y = X.filter(["Altezza di precipitazione (m)"])
    y = y.rename(columns={"Altezza di precipitazione (m)": "Altezza"})
    X = X.drop(["Unnamed: 0", "Altezza di precipitazione (m)", "Casi", "mgh"], axis=1)

    if dataset_settings.get("only_BMI") == 1:
        X = X.drop(["Peso", "Altezza soggetto"], axis=1)

    if dataset_settings.get("total_dmg"):
        def total_dmg(row):
            max_dmg = 0
            row["total_dmg"] = 0

            for k, v in no_injury_parts.items():
                row["total_dmg"] = row["total_dmg"] + row[k]
                max_dmg = max_dmg + v

            row["total_dmg"] = row["total_dmg"] / max_dmg

            return row

        X = X.apply(lambda row: total_dmg(row), axis=1)

    if dataset_settings.get("percentage"):
        for k, v in no_injury_parts.items():
            X[k] = X[k].apply(lambda x: x / v)

    if dataset_settings.get("type_of_labels") == "floors":
        heights = y.sort_values('Altezza', ascending=True).Altezza.unique()
        for i, v in enumerate(heights):
            y.Altezza = y.Altezza.replace(v, i + 1)

    dataset_variants = dataset_settings.get('dataset_variants')
    dataset_list = []
    for dataset_variant in dataset_variants:
        y_ = y.copy()
        if dataset_variant == "complete":
            X_ = X
        elif dataset_variant == "only_binary":
            if dataset_settings.get("total_dmg"):
                X_ = X.drop(["TESTA", "TORACE", "ADDOME", "SCHELETRO", "total_dmg"], axis=1)
            else:
                X_ = X.drop(["TESTA", "TORACE", "ADDOME", "SCHELETRO"], axis=1)

        elif dataset_variant == "only_totals":
            if dataset_settings.get("total_dmg"):
                X_ = X.filter(
                    ["Sesso", "Età", "Altezza soggetto", "Peso", "BMI", "TESTA", "TORACE", "ADDOME", "SCHELETRO",
                     "total_dmg"])
            else:
                X_ = X.filter(
                    ["Sesso", "Età", "Altezza soggetto", "Peso", "BMI", "TESTA", "TORACE", "ADDOME", "SCHELETRO"])
        else:
            raise Exception(f"{dataset_variant} is not an available variant of the dataset")

        binarize = dataset_settings.get("binarize")
        if binarize is not None:
            for b in binarize:
                y_ = y.copy()
                y_.loc[y_['Altezza'] <= b, 'Altezza'] = 0
                y_.loc[y_['Altezza'] > b, 'Altezza'] = 1
                if binarize is not None:
                    print(
                        f"NUM_EXAMPLES low floors vs high floors with threshold {b}: {len(y_.loc[y_['Altezza'] == 0])} // {len(y_.loc[y_['Altezza'] == 1])}")
                dataset_list.append({"dataset_variant": dataset_variant, "X": X_, "y": y_})
        else:
            dataset_list.append({"dataset_variant": dataset_variant, "X": X_, "y": y_})

    return dataset_list

@Cache(
    cache_path="{working_path}\\{result_path}\\{experiment_name}.json"
)
def experiment(working_path, result_path, experiment_name, experiment_setup, dataset_name = "victims_of_fall_V1"):
    path_str = f"{working_path}\\data\\{dataset_name}.csv"
    df = pd.read_csv(path_str)
    binarize = None if "binarize" not in experiment_setup.keys() else eval(experiment_setup["binarize"])

    dataset_list =  get_dataset(df, experiment_setup["dataset_settings"])

    all_performance = []
    for dataset in dataset_list:
        dataset_variant = dataset["dataset_variant"]
        print(dataset_variant)

        X = dataset["X"].to_numpy()
        y = dataset["y"].to_numpy(dtype='int32').ravel()



        # TYPE OF TASK
        task = experiment_setup['task']

        # THE MASK IS REQUIRED WHEN MASKEDPCA OR MASKEDSVD IS USED
        if experiment_setup["dataset_settings"]["only_BMI"] == 1:
            mask = np.arange(X.shape[1]) > 2
        else:
            mask = np.arange(X.shape[1]) > 4

        # BUILDING THE PIPELINE
        pipe_steps = []
        for key, value in experiment_setup["pipe"].items():
            pipe_steps.append((key, eval(value)))
        pipe = Pipeline(pipe_steps)

        hp_optimizer = experiment_setup["hp_optimizer"]
        metrics = hp_optimizer.get("metrics")



        params_list = []
        parsed_params = {}
        for parameter, values in hp_optimizer.get("params")[0].items():
            parsed_params[parameter] = eval(values)
        params_list.append(parsed_params)

        list_skf = []

        if "n_split_outer_cv" in hp_optimizer.keys():
            cv_type = "n_split_outer_cv"
            list_skf.append(StratifiedKFold(n_splits=hp_optimizer[cv_type], shuffle=True, random_state=42))

        if "n_split_inner_cv" in hp_optimizer.keys():
            cv_type = "n_split_inner_cv"
            list_skf.append(StratifiedKFold(n_splits=hp_optimizer[cv_type], shuffle=True, random_state=42))

        if hp_optimizer["type"] == 'GridSearchCV':
            optimizer = GridSearchCV(pipe, parsed_params, n_jobs=None, cv=list_skf[-1], verbose=1, scoring=metrics,
                                     refit=metrics[0], return_train_score=True,error_score="raise").fit(X, y)
        elif hp_optimizer["type"] == 'RandomizeSearchCV':
            n_iter = hp_optimizer["n_iter"] if "n_iter" in hp_optimizer.keys() else 100
            print(f"n_iter:{n_iter}")
            optimizer = RandomizedSearchCV(pipe, parsed_params, n_iter=n_iter, cv=list_skf[-1], verbose=1,
                                           scoring=metrics, refit=metrics[0], return_train_score=True).fit(X, y)
        elif hp_optimizer["type"] == 'BayesSearchCV':
            n_iter = hp_optimizer["n_iter"] if "n_iter" in hp_optimizer.keys() else 100
            print(f"n_iter:{n_iter}")
            optimizer = BayesSearchCV(pipe, parsed_params, n_jobs=cpu_1, n_iter=n_iter, cv=list_skf[-1], verbose=1,
                                      scoring=metrics, refit=metrics[0], return_train_score=True).fit(X, y)

        if "n_split_inner_cv" in hp_optimizer.keys():
            cv_dic = cross_validate(optimizer, X, y, cv=list_skf[0], n_jobs=None, scoring=metrics,
                                    return_estimator=True, verbose=2, return_train_score=True)
            best_params_cv = [estimator.best_params_ for estimator in cv_dic["estimator"]]

            scores_test_dict = {}
            scores_train_dict = {}
            for metric in metrics:
                scores_test_dict[metric] = np.mean(cv_dic[f"test_{metric}"])
                scores_train_dict[metric] = np.mean(cv_dic[f"train_{metric}"])

            cv_results = str(cv_dic)
        else:
            best_params_cv = [optimizer.best_params_]
            best_model = pd.DataFrame(optimizer.cv_results_).iloc[optimizer.best_index_]

            scores_test_dict = {}
            scores_train_dict = {}
            for metric in metrics:
                scores_test_dict[metric] = best_model[f"mean_test_{metric}"]
                scores_train_dict[metric] = best_model[f"mean_train_{metric}"]

            cv_results = str(optimizer.cv_results_)

        score = {
            "experiment_name": experiment_name,
            "dataset_variant": dataset_variant,
            "estimator": experiment_setup['pipe']['estimator'],
            "task": task,
            "hp_optimizer": hp_optimizer['type'],
            "cv_type": cv_type,
            "mean_test_score": scores_test_dict,
            "mean_train_score": scores_train_dict,
            "best_params": str(best_params_cv),
            "cv_results": cv_results,
            "experiment_setup": experiment_setup
        }

        all_performance.append(score)

    return all_performance

def get_holdout_predictions(working_path, result_path, experiment_name, experiment_setup, dataset_name = "victims_of_fall_V1"):
    path_str = f"{working_path}\\data\\{dataset_name}.csv"
    df = pd.read_csv(path_str)
    binarize = None if "binarize" not in experiment_setup.keys() else eval(experiment_setup["binarize"])

    dataset_list = get_dataset(df, experiment_setup["dataset_settings"])

    predictions = []
    for dataset in dataset_list:
        dataset_variant = dataset["dataset_variant"]
        print(dataset_variant)

        X = dataset["X"].to_numpy()
        y = dataset["y"].to_numpy(dtype='int32').ravel()

        # TYPE OF TASK
        task = experiment_setup['task']

        # THE MASK IS REQUIRED WHEN MASKEDPCA OR MASKEDSVD IS USED
        if experiment_setup["dataset_settings"]["only_BMI"] == 1:
            mask = np.arange(X.shape[1]) > 2
        else:
            mask = np.arange(X.shape[1]) > 4

        # BUILDING THE PIPELINE
        pipe_steps = []
        for key, value in experiment_setup["pipe"].items():
            print((key,value))
            pipe_steps.append((key, eval(value)))
        pipe = Pipeline(pipe_steps)

        sss = StratifiedShuffleSplit(n_splits=1,  random_state=42)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            predictions.append((X_test, y_test, pred))

    print(predictions)
    score = {
        "X_test": predictions[0][0],
        "y_test": predictions[0][1],
        "pred": predictions[0][2],
        "task": task,
        "experiment_setup": experiment_setup
    }
    return score

