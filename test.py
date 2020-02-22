"""
Evaluate our performance on test data
"""

from utils import *
import xlwt
from sklearn.metrics import matthews_corrcoef
from xlrd import open_workbook
import matplotlib.pyplot as plt
import argparse
import os

def all_train(train_x_orig, train_y_orig, train_x_phred, train_y_phred):
    ## Get the two models (using original and Phred) trained on all samples
    print("Training on whole original dataset")
    print("Training random forest classifier")
    model_rf_orig = RandomForestClassifier(n_estimators=2000)
    model_rf_orig.fit(train_x_orig, train_y_orig)
    joblib.dump(model_rf_orig, "model/rf_orig_all.pkl")

    print("Training XGBoost classifier")
    model_xgbt_orig = XGBClassifier(colsample_bytree=0.8, colsample_bylevel=0.6, scale_pos_weight=3.0,
                               learning_rate=0.1, n_estimators=2000, subsample=1.0, reg_alpha=0.0, reg_lambda=0.01,
                               max_depth=16, gamma=0.0, min_child_weight=0, booster='gbtree')
    model_xgbt_orig.fit(train_x_orig, train_y_orig)
    joblib.dump(model_xgbt_orig, "model/xgboost_orig_all.pkl")

    ### Phred
    print("Training on whole Phred dataset")
    print("Training random forest classifier")
    model_rf_phred = RandomForestClassifier(n_estimators=5000)
    model_rf_phred.fit(train_x_orig, train_y_orig)
    joblib.dump(model_rf_phred, "model/rf_phred_all.pkl")

    print("Training XGBoost classifier")
    model_xgbt_phred = XGBClassifier(colsample_bytree=0.6, colsample_bylevel=1.0, scale_pos_weight=3.0,
                               learning_rate=0.1, n_estimators=4000, subsample=0.8, reg_alpha=0.01, reg_lambda=0.1,
                               max_depth=23, gamma=3.6, min_child_weight=0, booster='gbtree')
    model_xgbt_phred.fit(train_x_phred, train_y_phred)
    joblib.dump(model_xgbt_phred, "model/xgboost_phred_all.pkl")

def cleaned_train(train_x_orig, train_y_orig, train_x_phred, train_y_phred):
    ## Get the two models (using original and Phred) trained on cleaned samples
    print("Training on cleaned orgiginal dataset")
    print("Training random forest classifier")
    model_rf_orig = RandomForestClassifier(n_estimators=2000)
    model_rf_orig.fit(train_x_orig, train_y_orig)
    joblib.dump(model_rf_orig, "model/rf_orig_cleaned.pkl")

    print("Training XGBoost classifier")
    model_xgbt_orig = XGBClassifier(colsample_bytree=0.8, colsample_bylevel=0.5, scale_pos_weight=5.0,
                               learning_rate=0.1, n_estimators=3000, subsample=1.0, reg_alpha=0.001, reg_lambda=0.01,
                               max_depth=9, gamma=0.2, min_child_weight=0, booster='gbtree')
    model_xgbt_orig.fit(train_x_orig, train_y_orig)
    joblib.dump(model_xgbt_orig, "model/xgboost_orig_cleaned.pkl")

    ### Phred
    print("Training on cleaned Phred dataset")
    print("Training random forest classifier")
    model_rf_phred = RandomForestClassifier(n_estimators=5000)
    model_rf_phred.fit(train_x_orig, train_y_orig)
    joblib.dump(model_rf_phred, "model/rf_phred_cleaned.pkl")

    print("Training XGBoost classifier")
    model_xgbt_phred = XGBClassifier(colsample_bytree=0.4, colsample_bylevel=0.8, scale_pos_weight=8.0,
                            learning_rate=0.05, n_estimators=5000, subsample=0.8, reg_alpha=0.1, reg_lambda=0.01,
                            max_depth=29, gamma=0.0, min_child_weight=0, booster='gbtree')
    model_xgbt_phred.fit(train_x_phred, train_y_phred)
    joblib.dump(model_xgbt_phred, "model/xgboost_phred_cleaned.pkl")

def test_and_evaluate(test_x, test_y, model):
    y_prob = model.predict_proba(test_x)
    y_result = model.predict(test_x)
    print ("Accuracy:", accuracy_score(test_y, y_result))
    print ("Precision: ", precision_score(test_y, y_result))
    print ("Recall: ", recall_score(test_y, y_result))
    print ("F1 Score: ", f1_score(test_y, y_result))
    print("AUC: ", roc_auc_score(test_y, y_prob[:, -1]))

def save_xls(test_x, test_y, model, label_path, file_name):
    y_prob = model.predict_proba(test_x)
    _, tpr, _ = roc_curve(test_y, y_prob[:, -1])
    labeling = np.load(label_path, allow_pickle=True)
    title = ["Chr", "Start", "End", "Ref", "Alt", "Gene_system", "region", "Gene_symbol", "Effect", "Mutation_type", "AA_change", "Cytoband", "False", "True"]
    workbook = xlwt.Workbook(encoding="utf-8")
    booksheet = workbook.add_sheet("sheet1", cell_overwrite_ok=True)
    for i in range(len(title)):
        booksheet.write(0, i, title[i])
    for i in range(labeling.shape[0]):
        for j in range(labeling.shape[1]):
            if j == 7 or j == 10:
                booksheet.write(i+1, j, labeling[i][j][1:-1])
            else:
                booksheet.write(i + 1, j, labeling[i][j])
    for i in range(y_prob.shape[0]):
        booksheet.write(i+1, 12, float(y_prob[i][0]))
        booksheet.write(i+1, 13, float(y_prob[i][1]))
    workbook.save(file_name)

def save_csv(test_x, test_y, model, label_path, file_name, driverfile, nondriverfile):
    y_prob = model.predict_proba(test_x)
    y_result = model.predict(test_x)
    labeling = np.load(label_path, allow_pickle=True)
    title = np.array([["Chr", "Start", "End", "Ref", "Alt", "Gene_system", "region", "Gene_symbol", "Effect", "Mutation_type",
             "AA_change", "Cytoband", "False", "True"]])
    result = np.concatenate((labeling, y_prob), axis=1)
    result = np.concatenate((title, result), axis=0)
    np.savetxt(file_name, result, delimiter="\t", fmt='%s')

    pos_label = np.where(y_result==0)[0]
    neg_label = np.where(y_result==1)[0]
    new_title = np.array([["Chr", "Start", "End", "Ref", "Alt", "Gene_system", "region", "Gene_symbol", "Effect", "Mutation_type",
             "AA_change", "Cytoband"]])
    np.savetxt(driverfile, np.concatenate((new_title, labeling[pos_label])), delimiter="\t", fmt="%s")
    np.savetxt(nondriverfile, np.concatenate((new_title, labeling[neg_label])), delimiter="\t", fmt="%s")


def train():
    ## Training and saving model, this part only need to be done once
    ## Train on whole training dataset
    print("Train on whole training dataset and save model")
    dataset_orig = np.load("DriverBase/Orig_Data.npy")
    dataset_phred = np.load("DriverBase/Phred_Data.npy")

    dataset_orig_x, dataset_orig_y = dataset_orig[:, :-1], dataset_orig[:, -1].astype(np.int32)
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_orig_x = np.delete(dataset_orig_x, [23, 24, 25], axis=1)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    all_train(dataset_orig_x, dataset_orig_y, dataset_phred_x, dataset_phred_y)

    ## Train on cleaned training dataset
    print("Train on cleaned training dataset and save model")
    dataset_orig = np.load("DriverBase/cleaned_data_orig.npy")
    dataset_phred = np.load("DriverBase/cleaned_data_phred.npy")

    dataset_orig_x, dataset_orig_y = dataset_orig[:, :-1], dataset_orig[:, -1].astype(np.int32)
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_orig_x = np.delete(dataset_orig_x, [23, 24, 25], axis=1)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    cleaned_train(dataset_orig_x, dataset_orig_y, dataset_phred_x, dataset_phred_y)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--train_flag", default="True", help="Perform training or not. Only need to train the first time, if models already saved, set this to False")
    parser.add_argument("-d", "--data_type", default="orig", help="Data type of the test data")
    parser.add_argument("-tp", "--test_path", default="Test_Data_Final/Pancancer/Orig_Data.npy", help="Test data type")
    parser.add_argument("-of", "--output_folder", default="Test_Data_Final/Pancancer/")
    parser.add_argument("-lp", "--label_path", default="DriverBase/Orig_Label.npy")
    parser.add_argument("-l", "--labelExist", default="False", help="Whether label for test data exists. Set it to True if you are using test data and its label exists.")
    args = parser.parse_args()
    train_f = args.train_flag
    d_type = args.data_type
    test_path = args.test_path
    output_path = args.output_folder
    label_path = args.label_path
    label_exist = args.labelExist
    ## Training process
    if train_f == "True":
        train()

    ## Testing process
    dataset = np.load(test_path)
    dataset_x, dataset_y = dataset[:, :-1], dataset[:, -1].astype(np.int32)
    dataset_x = np.delete(dataset_x, [23, 24, 25], axis=1)
    if d_type == "orig":
        print("Test on original dataset")
        print("Testing on XGBoost trained with whole training set")
        model_xgbt_all = joblib.load("model/xgboost_orig_all.pkl")
        if label_exist == "True":
            test_and_evaluate(dataset_x, dataset_y, model_xgbt_all)
        #save_xls(dataset_x, dataset_y, model_xgbt_all, label_path, output_path+"xgboost_orig.xls")
        save_csv(dataset_x, dataset_y, model_xgbt_all, label_path, os.path.join(output_path, "xgboost_orig.csv"),
                 os.path.join(output_path, "xgboost_orig_driver.csv"), os.path.join(output_path, "xgboost_orig_passenger.csv"))

        print("Testing on XGBoost trained with cleaned training set")
        model_xgbt_cleaned = joblib.load("model/xgboost_orig_cleaned.pkl")
        if label_exist == "True":
            test_and_evaluate(dataset_x, dataset_y, model_xgbt_cleaned)
        #save_xls(dataset_x, dataset_y, model_xgbt_cleaned, label_path, output_path+"xgboost_orig_cleaned.xls")
        save_csv(dataset_x, dataset_y, model_xgbt_all, label_path, os.path.join(output_path, "xgboost_orig_cleaned.csv"),
                 os.path.join(output_path, "xgboost_orig_cleaned_driver.csv"), os.path.join(output_path, "xgboost_orig_cleaned_passenger.csv"))

    elif d_type == "phred":
        print("Testing on Phred dataset")
        print("Testing on XGBoost trained with whole training set")
        model_xgbt_all = joblib.load("model/xgboost_phred_all.pkl")
        if label_exist == "True":
            test_and_evaluate(dataset_x, dataset_y, model_xgbt_all)
        #save_xls(dataset_x, dataset_y, model_xgbt_all, label_path, output_path+"xgboost_phred.xls")
        save_csv(dataset_x, dataset_y, model_xgbt_all, label_path, os.path.join(output_path, "xgboost_phred.csv"),
                 os.path.join(output_path, "xgboost_phred_driver.csv"), os.path.join(output_path, "xgboost_phred_passenger.csv"))

        print("Testing on XGBoost trained with cleaned training set")
        model_xgbt_cleaned = joblib.load("model/xgboost_phred_cleaned.pkl")
        if label_exist == "True":
            test_and_evaluate(dataset_x, dataset_y, model_xgbt_cleaned)
        #save_xls(dataset_x, dataset_y, model_xgbt_cleaned, label_path, output_path+"xgboost_phred_cleaned.xls")
        save_csv(dataset_x, dataset_y, model_xgbt_all, label_path, os.path.join(output_path, "xgboost_phred_cleaned.csv"),
                 os.path.join(output_path, "xgboost_phred_cleaned_driver.csv"), os.path.join(output_path, "xgboost_phred_cleaned_passenger.csv"))
    else:
        print("Data type not allowed Refer to --help for more information")

