"""
Evaluate our performance on test data
"""

from utils import *
import xlwt
import matplotlib.pyplot as plt

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
    #fpr, tpr, _ = roc_curve(test_y, y_prob[:, -1])


def save_xls(test_x, model, file_name):
    y_prob = model.predict_proba(test_x)
    workbook = xlwt.Workbook(encoding="utf-8")
    booksheet = workbook.add_sheet("sheet1", cell_overwrite_ok=True)
    booksheet.write(0, 0, "false")
    booksheet.write(0, 1, "true")
    for i in range(y_prob.shape[0]):
        booksheet.write(i+1, 0, float(y_prob[i][0]))
        booksheet.write(i+1, 1, float(y_prob[i][1]))
    workbook.save("predict_result/"+file_name)


if __name__=="__main__":
    """
    ## Train on whole training dataset
    dataset_orig = np.load("Final_Dataset/Orig_Data.npy")
    dataset_phred = np.load("Final_Dataset/Phred_Data.npy")

    dataset_orig_x, dataset_orig_y = dataset_orig[:, :-1], dataset_orig[:, -1].astype(np.int32)
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_orig_x = np.delete(dataset_orig_x, [23, 24, 25], axis=1)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    all_train(dataset_orig_x, dataset_orig_y, dataset_phred_x, dataset_phred_y)

    ## Train on cleaned training dataset
    dataset_orig = np.load("Final_Dataset/cleaned_data_orig.npy")
    dataset_phred = np.load("Final_Dataset/cleaned_data_phred.npy")

    dataset_orig_x, dataset_orig_y = dataset_orig[:, :-1], dataset_orig[:, -1].astype(np.int32)
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_orig_x = np.delete(dataset_orig_x, [23, 24, 25], axis=1)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    cleaned_train(dataset_orig_x, dataset_orig_y, dataset_phred_x, dataset_phred_y)
    """
    
    ## Test on Pancancer
    dataset_orig = np.load("Test_Data_Final/Pancancer/Orig_Data_no_shuffle.npy")
    dataset_phred = np.load("Test_Data_Final/Pancancer/Phred_Data_no_shuffle.npy")

    dataset_orig_x, dataset_orig_y = dataset_orig[:, :-1], dataset_orig[:, -1].astype(np.int32)
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_orig_x = np.delete(dataset_orig_x, [23, 24, 25], axis=1)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    print("Test on original Pancancer dataset")
    print("Testing on XGBoost trained with whole training set")
    model_xgbt_all = joblib.load("model/xgboost_orig_all.pkl")
    #test_and_evaluate(dataset_orig_x, dataset_orig_y, model_xgbt_all)
    save_xls(dataset_orig_x, model_xgbt_all, "xgboost_orig_all_no_shuffle.xls")
    print("Testing on random forest trained with whole training set")
    model_rf_all = joblib.load("model/rf_orig_all.pkl")
    #test_and_evaluate(dataset_orig_x, dataset_orig_y, model_rf_all)
    save_xls(dataset_orig_x, model_rf_all, "rf_orig_all_no_shuffle.xls")
    print("Testing on XGBoost trained with cleaned training set")
    model_xgbt_cleaned = joblib.load("model/xgboost_orig_cleaned.pkl")
    #test_and_evaluate(dataset_orig_x, dataset_orig_y, model_xgbt_cleaned)
    save_xls(dataset_orig_x, model_xgbt_cleaned, "xgboost_orig_cleaned_no_shuffle.xls")
    print("Testing on random forest trained with cleaned training set")
    model_rf_cleaned = joblib.load("model/rf_orig_cleaned.pkl")
    #test_and_evaluate(dataset_orig_x, dataset_orig_y, model_rf_cleaned)
    save_xls(dataset_orig_x, model_rf_cleaned, "rf_orig_cleaned_no_shuffle.xls")

    print("Testing on Phred Pancancer dataset")
    print("Testing on XGBoost trained with whole training set")
    model_xgbt_all = joblib.load("model/xgboost_phred_all.pkl")
    #test_and_evaluate(dataset_phred_x, dataset_phred_y, model_xgbt_all)
    save_xls(dataset_phred_x, model_xgbt_all, "xgboost_phred_all_no_shuffle.xls")
    print("Testing on random forest trained with whole training set")
    model_rf_all = joblib.load("model/rf_phred_all.pkl")
    #test_and_evaluate(dataset_phred_x, dataset_phred_y, model_rf_all)
    save_xls(dataset_phred_x, model_rf_all, "rf_phred_all_no_shuffle.xls")
    print("Testing on XGBoost trained with cleaned training set")
    model_xgbt_cleaned = joblib.load("model/xgboost_phred_cleaned.pkl")
    #test_and_evaluate(dataset_phred_x, dataset_phred_y, model_xgbt_cleaned)
    save_xls(dataset_phred_x, model_xgbt_cleaned, "xgboost_phred_cleaned_no_shuffle.xls")
    print("Testing on random forest trained with cleaned training set")
    model_rf_cleaned = joblib.load("model/rf_phred_cleaned.pkl")
    #test_and_evaluate(dataset_phred_x, dataset_phred_y, model_rf_cleaned)
    save_xls(dataset_phred_x, model_rf_cleaned, "rf_phred_cleaned_no_shuffle.xls")
    """
    
    ## Test on BRCA1
    dataset_orig = np.load("Test_Data_Final/BRCA1/Orig_Data.npy")
    dataset_phred = np.load("Test_Data_Final/BRCA1/Phred_Data.npy")

    dataset_orig_x, dataset_orig_y = dataset_orig[:, :-1], dataset_orig[:, -1].astype(np.int32)
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_orig_x = np.delete(dataset_orig_x, [23, 24, 25], axis=1)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    print("Test on original BRCA1 dataset")
    print("Testing on XGBoost trained with whole training set")
    model_xgbt_all = joblib.load("model/xgboost_orig_all.pkl")
    test_and_evaluate(dataset_orig_x, dataset_orig_y, model_xgbt_all)
    print("Testing on random forest trained with whole training set")
    model_rf_all = joblib.load("model/rf_orig_all.pkl")
    test_and_evaluate(dataset_orig_x, dataset_orig_y, model_rf_all)
    print("Testing on XGBoost trained with cleaned training set")
    model_xgbt_cleaned = joblib.load("model/xgboost_orig_cleaned.pkl")
    test_and_evaluate(dataset_orig_x, dataset_orig_y, model_xgbt_cleaned)
    print("Testing on random forest trained with cleaned training set")
    model_rf_cleaned = joblib.load("model/rf_orig_cleaned.pkl")
    test_and_evaluate(dataset_orig_x, dataset_orig_y, model_rf_cleaned)

    print("Testing on Phred BRCA1 dataset")
    print("Testing on XGBoost trained with whole training set")
    model_xgbt_all = joblib.load("model/xgboost_phred_all.pkl")
    test_and_evaluate(dataset_phred_x, dataset_phred_y, model_xgbt_all)
    print("Testing on random forest trained with whole training set")
    model_rf_all = joblib.load("model/rf_phred_all.pkl")
    test_and_evaluate(dataset_phred_x, dataset_phred_y, model_rf_all)
    print("Testing on XGBoost trained with cleaned training set")
    model_xgbt_cleaned = joblib.load("model/xgboost_phred_cleaned.pkl")
    test_and_evaluate(dataset_phred_x, dataset_phred_y, model_xgbt_cleaned)
    print("Testing on random forest trained with cleaned training set")
    model_rf_cleaned = joblib.load("model/rf_phred_cleaned.pkl")
    test_and_evaluate(dataset_phred_x, dataset_phred_y, model_rf_cleaned)
    """