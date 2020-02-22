"""
The original xls file cannot be read by pandas, thus reading and changing to proper xls file is needed
Files with "training" are original files while the ones with "train" are converted files
"""

from __future__ import unicode_literals
import pandas as pd
from xlwt import Workbook
import io
import numpy as np
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer as Imputer
import argparse
import csv

def small_dataloader(dir, out_dir, transfer=False):
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

def dataloader(dir, out_dir, transfer=False):
    if transfer:
        filename = dir
        file1 = io.open(filename, "r", encoding="utf8")
        data = file1.readlines()
        f = open(out_dir, 'w')
        csv_writer = csv.writer(f)
        for i, row in enumerate(data):
            row_list = []
            for j, val in enumerate(row.replace('\n', '').split('\t')):
                row_list.append(val)
            csv_writer.writerow(row_list)
        f.close()
    tb = pd.read_csv(out_dir)
    return tb

def get_labels(table):
    """
    Get the label information of test data. This function is for outputing the xls files as mentioned in test.py
    :param table: table to be procecssed
    :return: numpy array
    """
    np_array = np.array(table)
    labeling = np_array[:, :12]
    return labeling

def table_to_npy(table):
    """
    Transfer table to npy, keep useful information and leave out useless ones
    The missing data are imputed by median, average, or constant
    The sequence of the data is not changed
    :param table: table to be processed
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

def get_test_dataset(dataset):
    """
    Get the dataset that has mixed labels so that we can train
    :param posData: type of np.array
    :param negData: type of np.array
    :return: shuffled whole dataset
    """
    return dataset

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pp", "--pathPositive", default="DriverBase/training_Y_orig.xlsx", help="Input positive data path")
    parser.add_argument("-pn", "--pathNegative", default="DriverBase/training_N_orig.xlsx", help="Input negative data path")
    parser.add_argument("-op", "--outPath", default="DriverBase/Orig_Data.npy", help="Ouput data path")
    parser.add_argument("-lp", "--labelPath", default="DriverBase/Orig_Label.npy", help="Test data label path")
    args = parser.parse_args()
    pathPos = args.pathPositive
    pathNeg = args.pathNegative
    output_path = args.outPath
    label_path = args.labelPath
    if pathNeg == "None":
        posTable = dataloader(pathPos, "DriverBase/trainY.csv", True)
        labels = get_labels(posTable)
        np.save(label_path, labels)
        pos_array = table_to_npy(posTable)
        orig_dataset = get_test_dataset(pos_array)
    else:
        posTable = dataloader(pathPos, "DriverBase/trainY.csv", True)  # Use True when first using the datasets, change to False for save of calculation
        negTable = dataloader(pathNeg, "DriverBase/trainN.csv", True)
        pos_array = table_to_npy(posTable)
        neg_array = table_to_npy(negTable)
        orig_dataset = get_dataset(pos_array, neg_array)
    np.save(output_path, orig_dataset)
