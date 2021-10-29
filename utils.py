"""utils file"""
from typing import List, Tuple

import numpy as np
import pandas as pd
import math


# data generation
def create_dataset(data_path: str) -> Tuple[List[str], List[int]]:
    dataset = pd.read_csv(data_path)
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # shuffle the dataset
    return list(dataset["sequences"]), list(dataset["label"])


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)



# calculate metrics 
def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    sensitivity = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn) / (tn + fp + 1e-06)
    mcc = float(tp * tn - fp * fn) / (
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06
    )
    return acc, sensitivity, specificity, mcc

