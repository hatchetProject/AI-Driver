"""
Define the basic training functions
Including gridsearchng for best parameters, saving the model, making predictions and calculating the scores
May include plotting ROC-AUC plots afterwards
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, accuracy_score
import numpy as np
from sklearn.externals import joblib

def train_svm_classifier(X, Y):
    """
    :param kernel: select from linear, poly, rbf, sigmoid
    :param C: penalty parameter C of the error term
    """
    print ("Training svm classifier")
    parameters = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "C": [0.01, 0.1, 1.0, 10.0],
    }
    clf = GridSearchCV(SVC(gamma="scale", probability=True), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
    Y = Y.astype(np.float64)
    print(X.dtype, Y.dtype)
    clf.fit(X, Y)
    print ("Training finished")
    print ("The best parameters: ", clf.best_params_)
    y_result = clf.predict(X)
    y_prob = clf.predict_proba(X)
    joblib.dump(clf, "model/svm_clf_mod.model")


def train_rfc_classifier(X, Y):
    """
    :param n_estimators: the number of estimators for the forest
    """
    print("Training random forest classifier")
    parameters = {
        "n_estimators": [20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000],
    }
    clf = GridSearchCV(RandomForestClassifier(), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
    clf.fit(X, Y)
    print("Training finished")
    print("The best parameters:", clf.best_params_)
    y_result = clf.predict(X)
    y_prob = clf.predict_proba(X)
    joblib.dump(clf, "model/rfc_clf_mod.model")


def train_adaboost_classifier(X, Y):
    """
    :param n_estimators: number of estimators
    :param learning_rate: learning rate
    :param algorithm: SAMME or SAMME.R, but SAMME.R typically better
    """
    print("Training Adabooost classifier")
    parameters = {
        "n_estimators": [20, 50, 100, 200, 500, 1000, 2000, 3000, 5000],
        "learning_rate": [0.01, 0.1, 1, 10],
        "algorithm": ["SAMME.R"],
    }
    clf = GridSearchCV(AdaBoostClassifier(), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
    clf.fit(X, Y)
    print("Training finished")
    print("The best parameters:", clf.best_params_)
    y_result = clf.predict(X)
    y_prob = clf.predict_proba(X)
    joblib.dump(clf, "model/adaboost_clf_mod.model")


def train_gbdt_classifier(X, Y):
    """
    :param n_estimators: number of estimators
    :param learning_rate: learning rate
    :param criterion: function of measuring the quality of a split:friedman_mse (generally better), mse, mae
    """
    print("Training Gradient Boosting Decision Tree")
    parameters = {
        "n_estimators": [1000, 2000, 5000],
        "learning_rate": [0.01, 0.1, 1, 10],
        "criterion": ["mse"]
    }
    clf = GridSearchCV(GradientBoostingClassifier(), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
    clf.fit(X, Y)
    print("Training finished")
    print("The best parameters:", clf.best_params_)
    y_result = clf.predict(X)
    y_prob = clf.predict_proba(X)
    joblib.dump(clf, "model/gbdt_clf_mod.model")


def train_vc_classifier(X, Y):
    """
    :param estimators: list of tuples, containing the name and classifier type
    :param voting: different prediction criterions: hard, soft
    """
    print("Training voting classifier")
    parameters = {
        "voting": ["soft"],
    }
    clf = GridSearchCV(VotingClassifier([("RF", RandomForestClassifier(n_estimators=1000, random_state=1)),
                        ("GBDT", GradientBoostingClassifier(n_estimators=1000))]),
                        param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
    clf.fit(X, Y)
    print("Training finished")
    print("The best parameters:", clf.best_params_)
    y_result = clf.predict(X)
    joblib.dump(clf, "model/vc_clf_mod.model")


def train_mlp_classifier(X, Y):
    """
    No explicit parameters, if you like to change the network, then change it here
    """
    print("Training MLP classifier")
    clf = MLPClassifier(hidden_layer_sizes=(50, 10, ), activation="relu", solver="adam",
                             alpha=1e-4, batch_size=64, learning_rate='constant', learning_rate_init=1e-3,
                             power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=1e-4,
                             verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                             early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                             epsilon=1e-8)
    clf.fit(X, Y)
    print("Training finished")
    y_result = clf.predict(X)
    y_prob = clf.predict_proba(X)
    joblib.dump(clf, "model/mlp_clf_mod.model")


def train_xgb_classifier(X, Y):
    """
    Including multiple parameters to be evaluated
    """
    print("Training XGBoost classifier")
    parameters = {
        "booster": ["gbtree"],
        "n_estimators": [100, 500, 1000, 2000],
        "learning_rate": [0.01, 0.05, 0.1],
        #"gamma": np.arange(0, 10, 0.2),
        "max_depth": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "reg_alpha": [1, 0.1, 0.01, 0.001, 0],
        "reg_lambda": [1, 0.1, 0.01, 0.001, 0],
        #"scale_pos_weight": np.arange(1, 11, 1),
        #"subsample": [0.8, 0.2, 0.4, 0.6, 1.0],
        #"colsample_bylevel": np.arange(0, 1.1, 0.1),
        #"colsample_bytree": np.arange(0, 1.2, 0.2),
        #"min_child_weight": np.arange(0, 10, 1),
    }
    clf = GridSearchCV(XGBClassifier(), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=10)
    clf.fit(X, Y)
    print("Training finished")
    print("The best parameters:", clf.best_params_)
    joblib.dump(clf, "model/xgb_clf_mod.model")


def get_feature_importance(model, X, Y):
    """
    Get the feature importance of different models.
    Allowed models include Random Forest, Gradient Boosting Tree, Adaboost, XGBoost
    :param dir: The directory of models
    :return: Feature importances
    """
    model.fit(X, Y)
    return model.feature_importances_

