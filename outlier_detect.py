"""
Detect the outliers of the data
Do it on original and Phred, see whether there are differences
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, accuracy_score
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

def cleaned_phred():
    # Remove the outliers in the dataset, cleaning is done based on different labels
    dataset_phred = np.load("Final_Dataset/Phred_Data.npy")
    dataset_phred_x, dataset_phred_y = dataset_phred[:, :-1], dataset_phred[:, -1].astype(np.int32)
    dataset_phred_x = np.delete(dataset_phred_x, [23, 24, 25], axis=1)
    pos_idx = np.where(dataset_phred_y == 1)[0]
    neg_idx = np.where(dataset_phred_y == 0)[0]

    ## Remove the outliers from the positive samples
    dataset_pos_x = dataset_phred_x[pos_idx]
    detector_pos = IsolationForest(n_estimators=200)
    judgement_phred = detector_pos.fit_predict(dataset_pos_x)
    inlier = np.where(judgement_phred == 1)[0]
    outlier = np.where(judgement_phred == -1)[0]
    new_data_pos_x = dataset_phred_x[inlier]
    new_data_pos_y = dataset_phred_y[inlier]
    new_data_pos_y = new_data_pos_y.reshape(-1, 1)
    new_data_pos = np.concatenate((new_data_pos_x, new_data_pos_y), axis=1)

    ## Remove the outliers from the negative samples
    dataset_neg_x = dataset_phred_x[neg_idx]
    detector_neg = IsolationForest(n_estimators=200)
    judgement_phred = detector_neg.fit_predict(dataset_neg_x)
    inlier = np.where(judgement_phred == 1)[0]
    outlier = np.where(judgement_phred == -1)[0]
    new_data_neg_x = dataset_phred_x[inlier]
    new_data_neg_y = dataset_phred_y[inlier]
    new_data_neg_y = new_data_neg_y.reshape(-1, 1)
    new_data_neg = np.concatenate((new_data_neg_x, new_data_neg_y), axis=1)

    ## Get the dataset with outliers removed
    new_data = np.concatenate((new_data_pos, new_data_neg), axis=0)
    np.random.shuffle(new_data)
    print(new_data.shape)
    np.save("Final_Dataset/cleaned_data_phred.npy", new_data)

def cleaned_orig():
    dataset_orig = np.load("Final_Dataset/Orig_Data.npy")
    dataset_orig_x, dataset_orig_y = dataset_orig[:, :-1], dataset_orig[:, -1].astype(np.int32)
    dataset_orig_x = np.delete(dataset_orig_x, [23, 24, 25], axis=1)
    pos_idx = np.where(dataset_orig_y == 1)[0]
    neg_idx = np.where(dataset_orig_y == 0)[0]

    ## Remove the outliers from the positive samples
    dataset_pos_x = dataset_orig_x[pos_idx]
    detector_pos = IsolationForest(n_estimators=200)
    judgement_phred = detector_pos.fit_predict(dataset_pos_x)
    inlier = np.where(judgement_phred == 1)[0]
    outlier = np.where(judgement_phred == -1)[0]
    new_data_pos_x = dataset_orig_x[inlier]
    new_data_pos_y = dataset_orig_y[inlier]
    new_data_pos_y = new_data_pos_y.reshape(-1, 1)
    new_data_pos = np.concatenate((new_data_pos_x, new_data_pos_y), axis=1)

    ## Remove the outliers from the negative samples
    dataset_neg_x = dataset_orig_x[neg_idx]
    detector_neg = IsolationForest(n_estimators=200)
    judgement_phred = detector_neg.fit_predict(dataset_neg_x)
    inlier = np.where(judgement_phred == 1)[0]
    outlier = np.where(judgement_phred == -1)[0]
    new_data_neg_x = dataset_orig_x[inlier]
    new_data_neg_y = dataset_orig_y[inlier]
    new_data_neg_y = new_data_neg_y.reshape(-1, 1)
    new_data_neg = np.concatenate((new_data_neg_x, new_data_neg_y), axis=1)

    ## Get the dataset with outliers removed
    new_data = np.concatenate((new_data_pos, new_data_neg), axis=0)
    np.random.shuffle(new_data)
    print(new_data.shape)
    np.save("Final_Dataset/cleaned_data_orig.npy", new_data)

if __name__=="__main__":
    #cleaned_phred()
    cleaned_orig()
