"""
Train the models, save the models, obtain the results
10-fold cross-validation is used
"""

import numpy as np
from utils import *
import argparse
from outlier_detect import cleaned_orig, cleaned_phred




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default="DriverBase/Orig_Data.npy", help="Dataset path")
    parser.add_argument("-m", "--method", default="xgboost",
                        help="Method to be used for training. 6 methods provided, including SVM, Random Forest, AdaBoost, MLP, Gradient Boosting Tree and XGBoost.")
    args = parser.parse_args()
    data_path = args.path
    method = args.method
    train_x, train_y = None, None
    dataset = np.load(data_path)
    dataset_x, dataset_y = dataset[:, :-1], dataset[:, -1].astype(np.int32)
    dataset_x = np.delete(dataset_x, [23, 24, 25], axis=1)
    train_x = dataset_x
    train_y = dataset_y
    print("Training data with shape:", train_x.shape)
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
        print("Parameters wrongly specified")