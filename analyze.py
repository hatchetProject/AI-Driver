"""
Do analysis on models and datasets
"""

import numpy as np
from utils import get_feature_importance
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost
from xgboost import XGBClassifier
import shap
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from scipy.stats import spearmanr
import joblib
import matplotlib
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import average_precision_score

def feature_importances_original(data_path, output_path="final_plot/importance_orig.png"):
    dataset_orig = np.load(data_path)
    dataset_orig_x, dataset_orig_y = dataset_orig[:, :-1], dataset_orig[:, -1].astype(np.int32)
    dataset_orig_x = np.delete(dataset_orig_x, [23, 24, 25], axis=1)

    ## Analyzed with no Exact features
    print ("Get the feature importance indicated by different models")
    print ("Feature importance by Random Forest:")
    model_rf = RandomForestClassifier(n_estimators=500)
    rf_importance = get_feature_importance(model_rf, dataset_orig_x, dataset_orig_y)
    print ("Feature impotance by XGBoost: ")
    model_xgbt = XGBClassifier(colsample_bytree=0.4, colsample_bylevel=0.7, scale_pos_weight=10.0,
                               learning_rate=0.1, n_estimators=2000, subsample=0.8, reg_alpha=0.1, reg_lambda=0.01,
                               max_depth=10, gamma=0.2, min_child_weight=0, booster='gbtree')
    xgbt_importance = get_feature_importance(model_xgbt, dataset_orig_x, dataset_orig_y)
    print ("Feature importance by Adaboost: ")
    model_ada = AdaBoostClassifier(n_estimators=2000, learning_rate=0.1, algorithm='SAMME.R')
    ada_importance = get_feature_importance(model_ada, dataset_orig_x, dataset_orig_y)
    print ("Feature importance by Gradient Boosting Tree: ")
    model_gbdt = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.1, criterion='mse')
    gbdt_importance = get_feature_importance(model_gbdt, dataset_orig_x, dataset_orig_y)

    # Plot graph
    bar_width = 0.3
    plt.bar(np.arange(0, 46, 2) - 1.5 * bar_width, rf_importance, label="Random Forest", color="blue", alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(0, 46, 2) - 0.5 * bar_width, gbdt_importance, label="Gradient Boosting Tree", color="green",
            alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(0, 46, 2) + 0.5 * bar_width, ada_importance, label="AdaBoost", color="red", alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(0, 46, 2) + 1.5 * bar_width, xgbt_importance, label="XGBoost", color="yellow", alpha=0.8,
            width=bar_width)
    plt.title("Feature Importance By Different Methods On Original Dataset")
    plt.xlabel("Feature Name")
    plt.ylabel("Importance Ratio")
    plt.xticks(np.arange(0, 46, 2), ("SIFT", "Poly_HDIV", "Poly_HVAR", "LRT", "MuTaster", "MuAssessor",
                                     "FATHMM", "PROVEAN", "VEST3", "MetaSVM", "MetaLR", "M-CAP", "CADD", "DANN",
                                     "fathmm-MKL",
                                     "Eigen", "GenoCanyon", "fitCons", "GERP++", "pP100way",
                                     "pC100way", "SiPhy", "REVEL"), rotation=30)
    plt.legend()
    #plt.savefig(output_path)
    plt.show()

def feature_importances_phred(data_path, output_path="final_plot/importance_orig.png"):
    dataset_phred = np.load(data_path)
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)

    ## Analyzed with 23 features
    print ("Get the feature importance indicated by different models")
    print ("Feature importance by Random Forest:")
    model_rf = RandomForestClassifier(n_estimators=2000)
    rf_importance = get_feature_importance(model_rf, dataset_phred_x, dataset_phred_y)
    print ("Feature impotance by XGBoost: ")
    model_xgbt = XGBClassifier(colsample_bytree=0.8, colsample_bylevel=0.7, scale_pos_weight=5.0,
                               learning_rate=0.1, n_estimators=2000, subsample=1.0, reg_alpha=0,  reg_lambda=0.1,
                               max_depth=10, gamma=0.2, min_child_weight=0, booster='gbtree')
    xgbt_importance = get_feature_importance(model_xgbt, dataset_phred_x, dataset_phred_y)
    print ("Feature importance by Adaboost: ")
    model_ada = AdaBoostClassifier(n_estimators=500, learning_rate=1, algorithm='SAMME.R')
    ada_importance = get_feature_importance(model_ada, dataset_phred_x, dataset_phred_y)
    print ("Feature importance by Gradient Boosting Tree: ")
    model_gbdt = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.1, criterion='mse')
    gbdt_importance = get_feature_importance(model_gbdt, dataset_phred_x, dataset_phred_y)

    # Plot graph
    bar_width = 0.3
    plt.bar(np.arange(0, 46, 2) - 1.5 * bar_width, rf_importance, label="Random Forest", color="blue", alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(0, 46, 2) - 0.5 * bar_width, gbdt_importance, label="Gradient Boosting Tree", color="green", alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(0, 46, 2) + 0.5 * bar_width, ada_importance, label="AdaBoost", color="red", alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(0, 46, 2) + 1.5 * bar_width, xgbt_importance, label="XGBoost", color="yellow", alpha=0.8,
            width=bar_width)
    plt.title("Feature Importance By Different Methods On Phred Dataset")
    plt.xlabel("Feature Name")
    plt.ylabel("Importance Ratio")
    plt.xticks(np.arange(0, 46, 2), ("SIFT", "Poly_HDIV", "Poly_HVAR", "LRT", "MuTaster", "MuAssessor",
                                     "FATHMM", "PROVEAN", "VEST3", "MetaSVM", "MetaLR", "M-CAP", "CADD", "DANN",
                                     "fathmm-MKL",
                                     "Eigen", "GenoCanyon", "fitCons", "GERP++", "pP100way",
                                     "pC100way", "SiPhy", "REVEL"), rotation=30)
    plt.legend()
    #plt.savefig(output_path)
    plt.show()

def roc_plot_orig(data_path):
    dataset_orig = np.load(data_path)
    dataset_orig_x, dataset_orig_y = dataset_orig[:, :-1], dataset_orig[:, -1].astype(np.int32)
    dataset_orig_x = np.delete(dataset_orig_x, [23, 24, 25], axis=1)
    train_x, test_x, train_y, test_y = train_test_split(dataset_orig_x, dataset_orig_y, test_size=0.1, random_state=0)

    ### Plot the ROC curve for different methods
    print("ROC curve by SVM: ")
    model_svm = SVC(kernel='rbf', C=10.0, gamma="scale", probability=True)
    model_svm.fit(train_x, train_y)
    y_prob = model_svm.predict_proba(test_x)
    y_result = model_svm.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1], pos_label=1)
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y==0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn/fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.figure()
    plt.plot(fpr, tpr, label="SVM")

    print("ROC curve by Random Forest: ")
    model_rf = RandomForestClassifier(n_estimators=2000)
    model_rf.fit(train_x, train_y)
    y_prob = model_rf.predict_proba(test_x)
    y_result = model_rf.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="Random Forest")

    print("ROC curve by Adaboost: ")
    model_ada = AdaBoostClassifier(n_estimators=2000, learning_rate=1, algorithm='SAMME.R')
    model_ada.fit(train_x, train_y)
    y_prob = model_ada.predict_proba(test_x)
    y_result = model_ada.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="AdaBoost")

    print("ROC curve by MLP: ")
    model_mlp = MLPClassifier(hidden_layer_sizes=(50, 10,), activation="relu", solver="adam",
                              alpha=1e-4, batch_size=64, learning_rate='constant', learning_rate_init=1e-3,
                              power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=1e-4,
                              verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                              early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                              epsilon=1e-8)
    model_mlp.fit(train_x, train_y)
    y_prob = model_mlp.predict_proba(test_x)
    y_result = model_mlp.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="MLP")

    print("ROC curve by Gradient Boosting Tree: ")
    model_gbdt = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.1, criterion='mse')
    model_gbdt.fit(train_x, train_y)
    y_prob = model_gbdt.predict_proba(test_x)
    y_result = model_gbdt.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print ("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="Gradient Boosting Tree")

    print ("ROC curve by XGBoost: ")
    # Original data
    #model_xgbt = XGBClassifier(colsample_bytree=0.8, colsample_bylevel=0.6, scale_pos_weight=3.0,
    #                           learning_rate=0.1, n_estimators=2000, subsample=1.0, reg_alpha=0.0, reg_lambda=0.01,
    #                           max_depth=16, gamma=0.0, min_child_weight=0, booster='gbtree')

    # Cleaned data
    model_xgbt = XGBClassifier(colsample_bytree=0.8, colsample_bylevel=0.5, scale_pos_weight=5.0,
                                learning_rate=0.1, n_estimators=3000, subsample=1.0, reg_alpha=0.001, reg_lambda=0.01,
                                max_depth=9, gamma=0.2, min_child_weight=0, booster='gbtree')
    model_xgbt.fit(train_x, train_y)
    y_prob = model_xgbt.predict_proba(test_x)
    y_result = model_xgbt.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="XGBoost")

    plt.tick_params(labelsize=24)
    plt.xlabel("False Positive Rate", fontdict={"weight":"normal", "size":24})
    plt.ylabel("True Positive Rate", fontdict={"weight":"normal", "size":24})
    plt.title("ROC curves for different classifiers on original dataset", fontdict={"weight":"normal", "size":24})
    plt.legend(shadow=True, fontsize="x-large")
    plt.show()

def roc_plot_phred(data_path):
    dataset_phred = np.load(data_path)
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    #dataset_phred_x, dataset_phred_y, _ = cleaned_phred()
    train_x, test_x, train_y, test_y = train_test_split(dataset_phred_x, dataset_phred_y, test_size=0.1, random_state=0)
    ### Plot the ROC curve for different methods
    plt.figure()

    print("ROC curve by SVM: ")
    model_svm = SVC(kernel='rbf', C=10.0, gamma="scale", probability=True)
    model_svm.fit(train_x, train_y)
    y_prob = model_svm.predict_proba(test_x)
    y_result = model_svm.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="SVM")


    print("ROC curve by Random Forest: ")
    model_rf = RandomForestClassifier(n_estimators=2000)
    model_rf.fit(train_x, train_y)
    y_prob = model_rf.predict_proba(test_x)
    y_result = model_rf.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="Random Forest")

    print("ROC curve by Adaboost: ")
    model_ada = AdaBoostClassifier(n_estimators=2000, learning_rate=1, algorithm='SAMME.R')
    model_ada.fit(train_x, train_y)
    y_prob = model_ada.predict_proba(test_x)
    y_result = model_ada.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="AdaBoost")

    print("ROC curve by MLP: ")
    model_mlp = MLPClassifier(hidden_layer_sizes=(50, 10, ), activation="relu", solver="adam",
                             alpha=1e-4, batch_size=64, learning_rate='constant', learning_rate_init=1e-3,
                             power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=1e-4,
                             verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                             early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                             epsilon=1e-8)
    model_mlp.fit(train_x, train_y)
    y_prob = model_mlp.predict_proba(test_x)
    y_result = model_mlp.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="MLP")

    print("ROC curve by Gradient Boosting Tree: ")
    model_gbdt = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.1, criterion='mse')
    model_gbdt.fit(train_x, train_y)
    y_prob = model_gbdt.predict_proba(test_x)
    y_result = model_gbdt.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print ("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="Gradient Boosting Tree")

    print ("ROC curve by XGBoost: ")
    # Original dataset
    #model_xgbt = XGBClassifier(colsample_bytree=0.6, colsample_bylevel=1.0, scale_pos_weight=3.0,
    #                           learning_rate=0.1, n_estimators=4000, subsample=0.8, reg_alpha=0.01, reg_lambda=0.1,
    #                           max_depth=23, gamma=3.6, min_child_weight=0, booster='gbtree')

    # Cleaned dataset
    model_xgbt = XGBClassifier(colsample_bytree=0.4, colsample_bylevel=0.8, scale_pos_weight=8.0,
                              learning_rate=0.05, n_estimators=5000, subsample=0.8, reg_alpha=0.1, reg_lambda=0.01,
                              max_depth=29, gamma=0.0, min_child_weight=0, booster='gbtree')

    model_xgbt.fit(train_x, train_y)
    y_prob = model_xgbt.predict_proba(test_x)
    y_result = model_xgbt.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    print("Sensitivity: ", recall_score(test_y, y_result))
    fp_tn = np.sum(test_y == 0)
    t1 = np.where(np.equal(y_result, np.zeros(y_result.shape[0])))[0]
    t2 = np.where(np.equal(test_y, np.zeros(test_y.shape[0])))[0]
    tn = 0
    for idx in t1:
        if idx in t2:
            tn += 1
    print("Specificity: ", tn / fp_tn)
    print("MCC: ", matthews_corrcoef(test_y, y_result))
    plt.plot(fpr, tpr, label="XGBoost")

    plt.tick_params(labelsize=24)
    plt.xlabel("False Positive Rate", fontdict={"weight":"normal", "size":24})
    plt.ylabel("True Positive Rate", fontdict={"weight":"normal", "size":24})
    plt.title("ROC curves for different classifiers on Phred dataset", fontdict={"weight":"normal", "size":24})
    plt.legend(shadow=True, fontsize="x-large")
    plt.show()

def feature_roc(data_path):
    """
    Plot the roc curve for different features. ROC curves can also be interpreted as the change of tpr and fpr under different thresholds
    :return: graph
    """
    feature_names = ["SIFT", "Poly_HDIV", "Poly_HVAR", "LRT", "MuTaster", "MuAssessor",
                                     "FATHMM", "PROVEAN", "VEST3", "MetaSVM", "MetaLR", "M-CAP", "CADD", "DANN",
                                     "fathmm-MKL",
                                     "Eigen", "GenoCanyon", "fitCons", "GERP++", "pP100way",
                                     "pC100way", "SiPhy", "REVEL"]
    dataset_phred = np.load(data_path)
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    score = 0
    for i in range(len(feature_names)):
        score = roc_auc_score(dataset_phred_y, dataset_phred_x[:, i])
        score = round(score, 3)
        fpr, tpr, thresholds = roc_curve(dataset_phred_y, dataset_phred_x[:, i])
        plt.plot(fpr, tpr, label=feature_names[i]+": "+str(score))
    plt.legend(loc=4)
    plt.tick_params(labelsize=24)
    plt.xlabel("False positive rate", fontdict={"weight":"normal", "size":24})
    plt.ylabel("True positive rate", fontdict={"weight":"normal", "size":24})
    plt.title("Feature ROC curves on XXX dataset", fontdict={"weight":"normal", "size":30})
    plt.show()

def shap_explain(data_path):
    ## Analyzed with no Exact features
    dataset_phred = np.load(data_path)
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    f_names = ["SIFT", "Poly_HDIV", "Poly_HVAR", "LRT", "MuTaster", "MuAssessor",
                "FATHMM", "PROVEAN", "VEST3", "MetaSVM", "MetaLR", "M-CAP", "CADD", "DANN",
                "fathmm-MKL",
                "Eigen", "GenoCanyon", "fitCons", "GERP++", "pP100way",
                "pC100way", "SiPhy", "REVEL"]
    dataset_x_table = pd.DataFrame(dataset_phred_x, columns=f_names)

    print("Get the feature importance from SHAP: ")

    print("SHAP Feature analysis of XGBoost")
    #model_xgboost = XGBClassifier(colsample_bytree=0.8, colsample_bylevel=0.6, scale_pos_weight=3.0,
    #                           learning_rate=0.1, n_estimators=2000, subsample=1.0, reg_alpha=0.0, reg_lambda=0.01,
    #                           max_depth=16, gamma=0.0, min_child_weight=0, booster='gbtree')
    model_xgboost = XGBClassifier(colsample_bytree=0.6, colsample_bylevel=1.0, scale_pos_weight=3.0,
                               learning_rate=0.1, n_estimators=4000, subsample=0.8, reg_alpha=0.01, reg_lambda=0.1,
                               max_depth=23, gamma=3.6, min_child_weight=0, booster='gbtree')  # max_delta_step: 3
    model_xgboost.fit(dataset_x_table, dataset_phred_y)
    explainer_xgb = shap.TreeExplainer(model_xgboost)
    shap_values = explainer_xgb.shap_values(dataset_phred_x)
    #print (shap_values.shape)
    #shap.force_plot(explainer_xgb.expected_value, shap_values[0, :], dataset_phred_x[0, :], matplotlib=True)
    shap.force_plot(explainer_xgb.expected_value, shap_values[:100], dataset_x_table[:100])
    shap.summary_plot(shap_values, dataset_x_table)
    shap.summary_plot(shap_values, dataset_x_table, plot_type="bar")
    shap_interaction_values = explainer_xgb.shap_interaction_values(dataset_x_table)
    print(shap_interaction_values.shape)
    shap.summary_plot(shap_interaction_values, dataset_x_table)

def feature_analysis(data_path):
    feature_names = ["SIFT", "Poly_HDIV", "Poly_HVAR", "LRT", "MuTaster", "MuAssessor",
                     "FATHMM", "PROVEAN", "VEST3", "MetaSVM", "MetaLR", "M-CAP", "CADD", "DANN",
                     "fathmm-MKL",
                     "Eigen", "GenoCanyon", "fitCons", "GERP++", "pP100way",
                     "pC100way", "SiPhy", "REVEL"]
    dataset = np.load(data_path)
    dataset_x, dataset_y = dataset[:, :-1], dataset[:, -1].astype(np.int32)
    dataset_x = np.delete(dataset_x, [23, 24, 25], axis=1)
    score, aupr_score = 0, 0
    for i in range(len(feature_names)):
        roc_score = roc_auc_score(dataset_y, dataset_x[:, i])
        roc_score = round(roc_score, 3)
        aupr_score = average_precision_score(dataset_y, dataset_x[:, i])
        plt.scatter(roc_score, aupr_score, s=np.ones(roc_score.shape)*256, marker="o",
                    edgecolors="w", linewidth=np.ones(roc_score.shape)*1, label=feature_names[i])
        if feature_names[i] == "pC100way" or feature_names[i] == "REVEL":
            plt.annotate(feature_names[i], xy=(roc_score, aupr_score), xytext=(roc_score, aupr_score - 0.006))
        elif feature_names[i] == "FATHMM":
            plt.annotate(feature_names[i], xy=(roc_score, aupr_score), xytext=(roc_score, aupr_score - 0.004))
        elif feature_names[i] == "PROVEAN":
            plt.annotate(feature_names[i], xy=(roc_score, aupr_score), xytext=(roc_score, aupr_score + 0.009))
        elif feature_names[i] == "SiPhy":
            plt.annotate(feature_names[i], xy=(roc_score, aupr_score), xytext=(roc_score, aupr_score + 0.008))
        elif feature_names[i] == "SIFT":
            plt.annotate(feature_names[i], xy=(roc_score, aupr_score), xytext=(roc_score - 0.012, aupr_score))
        else:
            plt.annotate(feature_names[i], xy=(roc_score, aupr_score), xytext=(roc_score, aupr_score))
    model_xgbt_cleaned = joblib.load("model/xgboost_phred_cleaned.pkl")
    y_prob = model_xgbt_cleaned.predict_proba(dataset_x)
    ai_roc_score = roc_auc_score(dataset_y, y_prob[:, -1])
    ai_aupr_score = average_precision_score(dataset_y, y_prob[:, -1])
    plt.scatter(ai_roc_score, ai_aupr_score, s=np.ones(roc_score.shape)*256, marker="o",
                edgecolors="w", linewidth=np.ones(roc_score.shape)*1, label="AI-Driver")
    plt.annotate("AI-Driver", xy=(ai_roc_score, ai_aupr_score), xytext=(ai_roc_score, ai_aupr_score))

    plt.grid(axis="both")
    plt.tick_params(labelsize=24)
    plt.xlabel("AUPR", fontdict={"weight": "normal", "size": 24})
    plt.ylabel("AUROC", fontdict={"weight": "normal", "size": 24})
    plt.legend()
    plt.title("XXX Phred", fontdict={"weight": "normal", "size": 24})
    plt.show()

def spearman_rank_cor(train_data_path):
    data = np.load(train_data_path)
    dataset_phred_x = data[:, :-1]
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    feature_names = ["SIFT", "Poly_HDIV", "Poly_HVAR", "LRT", "MuTaster", "MuAssessor",
                     "FATHMM", "PROVEAN", "VEST3", "MetaSVM", "MetaLR", "M-CAP", "CADD", "DANN",
                     "fathmm-MKL",
                     "Eigen", "GenoCanyon", "fitCons", "GERP++", "pP100way",
                     "pC100way", "SiPhy", "REVEL"]
    rho, pval = spearmanr(dataset_phred_x)
    print (rho.shape)
    im = plt.imshow(rho, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title("Spearman rank correlation coefficients between features", fontdict={"size":18})
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataType", default="orig", help="Input data type")
    parser.add_argument("-p", "--dataPath", default="DriverBase/Orig_Data.npy", help="Input data path")
    args = parser.parse_args()
    data_type = args.dataType
    data_path = args.dataPath
    if data_type == "orig":
        print("Original feature importance: ")
        feature_importances_original(data_path)
        print("Plot ROC curves on original dataset: ")
        roc_plot_orig(data_path)
        print("SHAP explanation: ")
        shap_explain(data_path)
        print("Feature ROC: ")
        feature_roc(data_path)
        print("Spearman rank correlation heatmap: ")
        spearman_rank_cor(data_path)
    elif data_type == "phred":
        print("Phred feature importance: ")
        feature_importances_phred(data_path)
        print("Plot ROC curves on Phred dataset: ")
        roc_plot_phred(data_path)
        print("SHAP explanation: ")
        shap_explain(data_path)
        print("Feature ROC: ")
        feature_roc(data_path)
        print("Spearman rank correlation heatmap: ")
        spearman_rank_cor(data_path)
    elif data_type == "test":
        print("Feature ROC: ")
        feature_roc(data_path)
        print("Dot map:")
        feature_analysis(data_path)
    else:
        print("Data type not allowed. Refer to --help for more information.")
