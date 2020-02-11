"""
Train the models, save the models, obtain the results
10-fold cross-validation is used
"""

import numpy as np
from utils import *
import argparse
from outlier_detect import cleaned_orig, cleaned_phred

dataset_orig = np.load("Final_Dataset/cleaned_data_orig.npy")
dataset_phred = np.load("Final_Dataset/cleaned_data_phred.npy")

dataset_orig_x, dataset_orig_y = dataset_orig[:, :-1], dataset_orig[:, -1].astype(np.int32)
dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
dataset_orig_x = np.delete(dataset_orig_x, [23, 24, 25], axis=1)
dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)

#dataset_phred_x, dataset_phred_y, _ = cleaned_phred()
#dataset_orig_x, dataset_orig_y, _ = cleaned_orig()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="orig", help="Dataset chosen to be used")
    parser.add_argument("-m", "--method", default="xgboost", help="Method to be used for training")
    args = parser.parse_args()
    dataset = args.dataset
    method = args.method
    train_x, train_y = None, None
    if dataset == "orig":
        train_x = dataset_orig_x
        train_y = dataset_orig_y
    elif dataset == "phred":
        train_x = dataset_phred_x
        train_y = dataset_phred_y
    print (train_x.shape)
    if method == "svm":
        train_svm_classifier(train_x, train_y)
    elif method == "rfc":
        train_rfc_classifier(train_x, train_y)
    elif method == "adaboost":
        train_adaboost_classifier(train_x, train_y)
    elif method == "gbdt":
        train_gbdt_classifier(train_x, train_y)
    elif method == "vc":
        train_vc_classifier(train_x, train_y)
    elif method == "mlp":
        train_mlp_classifier(train_x, train_y)
    elif method == "xgbt":
        train_xgb_classifier(train_x, train_y)
    else:
        print ("Parameters wrongly specified")