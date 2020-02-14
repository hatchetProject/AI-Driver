"""
Do analysis on models and datasets
"""
# Features' ROC values
# Original features' table

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
from outlier_detect import cleaned_orig, cleaned_phred

def feature_importances_original():
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dataset_orig = np.load("DiverBase/Orig_Data.npy")
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
    #plt.savefig("feature_importance_original_23.png")
    plt.show()

def feature_importances_phred():
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dataset_phred = np.load("DiverBase/Phred_Data.npy")
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    ## Analyzed with all features
    """print "Get the feature importance indicated by different models"
    print "Feature importance by Random Forest:"
    model_rf = RandomForestClassifier(n_estimators=1000)
    rf_importance = get_feature_importance(model_rf, dataset_phred_x, dataset_phred_y)
    print "Feature importance by Gradient Boosting Tree:"
    model_gbdt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, criterion="friedman_mse")
    gbdt_importance = get_feature_importance(model_gbdt, dataset_phred_x, dataset_phred_y)
    print("Feature importance by Adaboost:")
    model_ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1, algorithm="SAMME.R")
    ada_importance = get_feature_importance(model_ada, dataset_phred_x, dataset_phred_y)
    print("Feature impotance by XGBoost: ")
    model_xgbt = XGBClassifier(reg_alpha=0, colsample_bytree=1.0, colsample_bylevel=1.0, scale_pos_weight=10, learning_rate=0.01, n_estimators=2000, subsample=0.8, reg_lambda=1, max_depth=5, gamma=0.0, booster='gbtree')
    xgbt_importance = get_feature_importance(model_xgbt, dataset_phred_x, dataset_phred_y)
    
    # Plot graph
    bar_width = 0.3
    plt.bar(np.arange(0, 52, 2)-1.5*bar_width, rf_importance, label="Random Forest", color="blue", alpha=0.8, width=bar_width)
    plt.bar(np.arange(0, 52, 2)-0.5*bar_width, gbdt_importance, label="Gradient Boosting Tree", color="red", alpha=0.8, width=bar_width)
    plt.bar(np.arange(0, 52, 2)+0.5*bar_width, ada_importance, label="Adaboost",color="green", alpha=0.8, width=bar_width)
    plt.bar(np.arange(0, 52, 2)+1.5*bar_width, xgbt_importance, label="XGBoost", color="yellow", alpha=0.8, width=bar_width)
    plt.title("Feature Importance By Different Methods")
    plt.xlabel("Feature Name")
    plt.ylabel("Importance Ratio")
    plt.xticks(np.arange(0, 52, 2), ("SIFT", "Poly_HDIV", "Poly_HVAR", "LRT", "MuTaster", "MuAssessor",
                               "FATHMM", "PROVEAN", "VEST3", "MetaSVM", "MetaLR", "M-CAP", "CADD", "DANN", "fathmm-MKL",
                               "Eigen", "GenoCanyon", "fitCons", "GERP++", "pP100way",
                               "pC100way", "SiPhy", "REVEL", "ExAC",
                               "ExAC_nontcga", "Damaging"), rotation=30)
    plt.legend()
    plt.show()
    """

    ## Analyzed with no Exact features
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
    #plt.savefig("feature_importance_phred_23.png")
    plt.show()

def roc_plot_orig():
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dataset_orig = np.load("Final_Dataset/cleaned_data_orig.npy")
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
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
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
    plt.plot(fpr, tpr, label="AdaBoost")

    """
    print("ROC curve by Voting Classifier: ")
    model_vc = VotingClassifier([("RF", RandomForestClassifier(n_estimators=2000, random_state=1)),
                                 ("GBDT", GradientBoostingClassifier(n_estimators=2000))], voting="soft")
    model_vc.fit(train_x, train_y)
    y_prob = model_vc.predict_proba(test_x)
    y_result = model_vc.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    plt.plot(fpr, tpr, label="Voting Classifier")
    """

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
    plt.plot(fpr, tpr, label="XGBoost")

    plt.tick_params(labelsize=24)
    plt.xlabel("False Positive Rate", fontdict={"weight":"normal", "size":24})
    plt.ylabel("True Positive Rate", fontdict={"weight":"normal", "size":24})
    plt.title("ROC curves for different classifiers on cleaned original dataset", fontdict={"weight":"normal", "size":24})
    plt.legend(shadow=True, fontsize="x-large")
    plt.show()

def roc_plot_phred():
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dataset_phred = np.load("Final_Dataset/cleaned_data_phred.npy")
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
    plt.plot(fpr, tpr, label="AdaBoost")

    """
    print("ROC curve by Voting Classifier: ")
    model_vc = VotingClassifier([("RF", RandomForestClassifier(n_estimators=2000, random_state=1)),
                      ("GBDT", GradientBoostingClassifier(n_estimators=2000))], voting="soft")
    model_vc.fit(train_x, train_y)
    y_prob = model_vc.predict_proba(test_x)
    y_result = model_vc.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))
    fpr, tpr, theta = roc_curve(test_y, y_prob[:, -1])
    plt.plot(fpr, tpr, label="Voting Classifier")
    """

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
    plt.plot(fpr, tpr, label="XGBoost")

    plt.tick_params(labelsize=24)
    plt.xlabel("False Positive Rate", fontdict={"weight":"normal", "size":24})
    plt.ylabel("True Positive Rate", fontdict={"weight":"normal", "size":24})
    plt.title("ROC curves for different classifiers on cleaned Phred dataset", fontdict={"weight":"normal", "size":24})
    plt.legend(shadow=True, fontsize="x-large")
    plt.show()

def feature_roc_phred():
    """
    Plot the roc curve for different features. ROC curves can also be interpreted as the change of tpr and fpr under different thresholds
    :return: graph
    """
    import matplotlib
    import matplotlib.pyplot as plt
    feature_names = ["SIFT", "Poly_HDIV", "Poly_HVAR", "LRT", "MuTaster", "MuAssessor",
                                     "FATHMM", "PROVEAN", "VEST3", "MetaSVM", "MetaLR", "M-CAP", "CADD", "DANN",
                                     "fathmm-MKL",
                                     "Eigen", "GenoCanyon", "fitCons", "GERP++", "pP100way",
                                     "pC100way", "SiPhy", "REVEL"]
    dataset_phred = np.load("Final_Dataset/Phred_Data.npy")
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
    plt.title("Feature ROC curves on Phred dataset", fontdict={"weight":"normal", "size":24})
    plt.show()

def feature_roc_orig():
    """
    Plot the roc curve for different features. ROC curves can also be interpreted as the change of tpr and fpr under different thresholds
    :return: graph
    """
    import matplotlib
    import matplotlib.pyplot as plt
    feature_names = ["SIFT", "Poly_HDIV", "Poly_HVAR", "LRT", "MuTaster", "MuAssessor",
                                     "FATHMM", "PROVEAN", "VEST3", "MetaSVM", "MetaLR", "M-CAP", "CADD", "DANN",
                                     "fathmm-MKL",
                                     "Eigen", "GenoCanyon", "fitCons", "GERP++", "pP100way",
                                     "pC100way", "SiPhy", "REVEL"]
    dataset_phred = np.load("Final_Dataset/cleaned_data_orig.npy")
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
    plt.title("Feature ROC curves on cleaned original dataset", fontdict={"weight":"normal", "size":30})
    plt.show()

def shap_explain():
    ## Analyzed with no Exact features
    dataset_phred = np.load("Final_Dataset/Phred_Data.npy")
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
    """

    print("SHAP Feature analysis of Random Forest")
    model_rf = RandomForestClassifier(n_estimators=2000)
    model_rf.fit(dataset_x_table, dataset_phred_y)
    explainer_rf = shap.TreeExplainer(model_rf)
    #shap_values = explainer_rf.shap_values(dataset_x_table)[0]
    #shap.force_plot(explainer_rf.expected_value[0], shap_values[0, :], dataset_phred_x[0, :], matplotlib=True)
    #shap.force_plot(explainer_rf.expected_value[0], shap_values[:100], dataset_x_table[:100])
    #shap.summary_plot(shap_values, dataset_x_table)
    #shap.summary_plot(shap_values, dataset_x_table, plot_type="bar")

    shap_interaction_values = explainer_rf.shap_interaction_values(dataset_x_table)
    shap.summary_plot(shap_interaction_values, dataset_x_table)
    """

if __name__=="__main__":
    feature_importances_phred()
    feature_importances_original()
    roc_plot_orig()
    roc_plot_phred()
    shap_explain()
    feature_roc_phred()