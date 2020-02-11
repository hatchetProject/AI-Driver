"""
The original xls file cannot be read by pandas, thus reading and changing to proper xls file is needed
Files with "training" are original files while the ones with "train" are converted files
"""

from __future__ import unicode_literals
import pandas as pd
from xlwt import Workbook
import io
import numpy as np
from sklearn.preprocessing import Imputer

def dataloader(dir, out_dir, transfer=False):
    if transfer:
        filename = dir
        # Opening the file using 'utf8' encoding
        file1 = io.open(filename, "r", encoding="utf8")
        data = file1.readlines()
        # Creating a workbook object
        xldoc = Workbook()
        # Adding a sheet to the workbook object
        sheet = xldoc.add_sheet("Sheet1", cell_overwrite_ok=True)
        # Iterating and saving the data to sheet
        for i, row in enumerate(data):
            # Two things are done here
            # Removeing the '\n' which comes while reading the file using io.open
            # Getting the values after splitting using '\t'
            for j, val in enumerate(row.replace('\n', '').split('\t')):
                sheet.write(i, j, val)
        # Saving the file as an excel file
        xldoc.save(out_dir)

    tb = pd.read_excel(out_dir)
    return tb


def table_to_npy(table, npy):
    """
    Transfer table to npy, keep useful information and leave out useless ones
    The missing data are imputed by median, average, or constant
    The sequence of the data is not changed
    :param table: table to be processed
    :param npy: output dir
    :return: numpy array
    """
    np_array = np.array(table)
    np_array = np_array[:, -27:]
    impute_median = Imputer(missing_values=np.NaN, strategy="median")
    #impute_0 = Imputer(missing_values=np.NaN, strategy="constant", fill_value=0)
    #impute_mean = Imputer(missing_values=np.NaN, strategy="mean")
    for i in range(np_array.shape[0]):
        for j in range(np_array.shape[1]):
            if np_array[i][j] != "-":
                np_array[i][j] = float(np_array[i][j])
            else:
                np_array[i][j] = np.NaN
    np_array = impute_median.fit_transform(np_array)
    np.save(npy, np_array)
    return np_array

def get_dataset(posData, negData):
    """
    Get the dataset that has mixed labels so that we can train
    :param posData: type of np.array
    :param negData: type of np.array
    :return: shuffled whole dataset
    """
    dataset = np.concatenate((posData, negData))
    np.random.shuffle(dataset)
    print("Dataset shape:", dataset.shape)
    return dataset


if __name__=="__main__":
    """
    pathPos = r"Test_Data_Final/BRCA1/BRCA1_positive_orig.xls"
    pathNeg = r"Test_Data_Final/BRCA1/BRCA1_negative_orig.xls"
    posTable = dataloader(pathPos, "Test_Data_Final/BRCA1/trainY.xls", True) # Use True when first using the datasets, change to False for save of calculation
    negTable = dataloader(pathNeg, "Test_Data_Final/BRCA1/trainN.xls", True)
    pos_array = table_to_npy(posTable, "Test_Data_Final/BRCA1/trainY_orig.npy")
    neg_array = table_to_npy(negTable, "Test_Data_Final/BRCA1/trainN_orig.npy")
    orig_dataset = get_dataset(pos_array, neg_array)
    print(orig_dataset.shape)
    np.save("Test_Data_Final/BRCA1/Orig_Data.npy", orig_dataset)

    path_phred_pos = r"Test_Data_Final/BRCA1/BRCA1_positive_Phred.xls"
    path_phred_neg = r"Test_Data_Final/BRCA1/BRCA1_negative_Phred.xls"
    posTablePhred = dataloader(path_phred_pos, "Test_Data_Final/BRCA1/trainY.xls", True)
    negTablePhred = dataloader(path_phred_neg, "Test_Data_Final/BRCA1/trainN.xls", True)
    pos_array_phred = table_to_npy(posTablePhred, "Test_Data_Final/BRCA1/trainY_phred.npy")
    neg_array_phred = table_to_npy(negTablePhred, "Test_Data_Final/BRCA1/trainN_phred.npy")
    phred_dataset = get_dataset(pos_array_phred, neg_array_phred)
    print(phred_dataset.shape)
    np.save("Test_Data_Final/BRCA1/Phred_Data.npy", phred_dataset)
    """

    pathPos = r"Test_Data_Final/Pancancer/Pancancer_orig.xls"
    posTable = dataloader(pathPos, "Test_Data_Final/Pancancer/trainY.xls", True)
    pos_array = table_to_npy(posTable, "Test_Data_Final/Pancancer/trainY_orig.npy")
    orig_dataset = pos_array
    #np.random.shuffle(orig_dataset)
    print(orig_dataset.shape)
    np.save("Test_Data_Final/Pancancer/Orig_Data_no_shuffle.npy", orig_dataset)

    pathPos = r"Test_Data_Final/Pancancer/Pancancer_Phred.xls"
    posTable = dataloader(pathPos, "Test_Data_Final/Pancancer/trainY.xls", True)
    pos_array = table_to_npy(posTable, "Test_Data_Final/Pancancer/trainY_phred.npy")
    phred_dataset = pos_array
    #np.random.shuffle(phred_dataset)
    print(phred_dataset.shape)
    np.save("Test_Data_Final/Pancancer/Phred_Data_no_shuffle.npy", phred_dataset)
