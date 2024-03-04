import json
import pandas as pd
import numpy as np
from numpy import array

def charge_results(files):
    results = []
    for i, file in enumerate(files):
        with open(file, "r") as f:
            data = f.read().replace('\n', '')
            experiment_results = json.loads(data)

            if type(experiment_results) == dict:
                experiment_results = [experiment_results]
            results.append(pd.DataFrame(experiment_results))
    return pd.concat(results)


def filter_and_sort_results(results, fields_value, target_metric):
    for k, v in results.items():
        results = results.loc[results[k] == v]

    results = results.reset_index().drop('index', axis=1)
    results = results.sort_values(by='mean_test_score', axis=0, ascending=False,
                                  key=lambda x: x.apply(lambda y: y[target_metric]))
    return results


def score_results_exploding(results):
    # THIS IS USED TO EXPLODE THE DICT OF THE SCORES AS COLUMNS OF THE RESULTED FILTERED_DF
    values = {}
    df_table = results[:0]
    estimators = {}
    for index, record in results.iterrows():
        cv_result = record.cv_results

        i_scoring = cv_result.rfind("scoring")
        i_test = cv_result[i_scoring:].find("'test")
        cv_result = "{" + cv_result[i_scoring + i_test:]
        cv_result = eval(cv_result)

        for k, list_scores in cv_result.items():
            if values.get(k) == None:
                values[k] = []
            mean = np.round(np.mean(np.abs(list_scores)), 2)
            std = np.round(np.std(np.abs(list_scores)), 2)

            values[k].append(str(mean) + " (std. " + str(std) + ")")

    for k, v in values.items():
        results[k] = v

    values = {}
    for record in results.mean_test_score:
        for k, v in record.items():
            if values.get(k) == None:
                values[k] = []
            values[k].append(v)

    for k, v in values.items():
        results[k] = v
    return results


def add_floor_field(results):
    floor_thresholds = []
    for i, v in results.iterrows():
        v = v['experiment_setup']
        v = str(v).replace("'", "\"")
        experiment_setup = json.loads(v)
        b = experiment_setup["dataset_settings"]['binarize']
        results.at[i, 'Floor Threshold'] = str(b[0])
    return results
