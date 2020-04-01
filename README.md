# AI-Driver
AI-Driver (AI-based driver classifier)is an ensemble method for predicting the driver status of somatic missense mutations based on 23 pathogenicity features. Missense mutations are the most common protein-coding mutations found in cancer genomes and increasing number of missense mutations has been recognized as clinically actionable variants. AI-Driver outperforms its individual constituent prediction tools, as expected for ensemble methods. We have shown that AI-Driver consistently has the best overall performance as compared to existing methods, particularly for distinguishing driver mutations from uncommon neutral missense mutations with an AF below 1%. AI-Driver also outperforms existing cancer-specific methods for distinguishing driver mutations from passenger mutations. Therefore, AI-Driver can be used to prioritize the most likely driver mutations among the sea of rare somatic mutations that are increasingly discovered as sequencing studies expand in scale.

## Download
* Clone and fork code and from this repository.
* Training data is available in the DriverBase folder and 23 features of whole exome for any possible variant can be downloaded [here](http://159.226.67.237/sun/AI-Driver/download/varcards.main.Phred_scaled.xls.gz).

## Usage
### Environment Requirement
You can install the environment by using [conda](https://docs.anaconda.com/anaconda/install/).
```python
conda create -n python3 python=3 xgboost=0.90 scikit-learn=0.221 xlrd xlwt xlutils pandas shap 
```
where the used packages include
* Python 3.6
* Pandas (with xlrd>=0.9.0)
* Sklearn 0.22.1
* XGBoost 0.90
* xlrd
* xlwt
* SHAP
* xlutils


### Basic Usage
* Run DataLoader.py on original raw Excel (.xls) files. This step extracts the useful information from files and shuffles the dataset for better robustness.
* Then run train.py by indicating the specific dataset and method. We use 10-fold cross-validation to determine the best parameters for each method. The datasets include original version, original+cleaned version, Phred version, Phred+cleaned version.The source of dataset can be changed according to your preference. Also, the range of parameters to be selected can be founded and changed in utils.py.  
* Analysis and testing of the models can be done in analyze.py and test.py.
* outlier_detect.py uses Isolation Forest to detect outliers in data and remove them from the dataset. The new datasets are "cleaned".
* The basic usage is described as below:
```python
python DataLoader.py -pp POSITIVE_PATH -pn NEGATIVE_PATH -op OUTPUT_PATH -lp LABEL_PATH -l TEST_LABEL_EXIST
python outlier_detect.py -ip INPUT_PATH -op OUTPUT_PATH -t DATA_TYPE
python train.py -p DATA_PATH -m METHOD
python test.py -f TRAIN -d DATA_TYPE -tp TEST_PATH -of OUTPUT_FOLDER -lp LABEL_PATH -l TEST_LABEL_EXIST
python analyze.py -d DATA_FORM -p DATA_PATH
```
#### Introductions of the parameters:
* Running DataLoader.py transforms the original xls files into npy file for continuous experiments. Missing value imputation 
and random shuffling of data is also done. POSITIVE_PATH indicates the path for positive training samples (for example: DriverBase/training_Y_orig.xls), 
NEGATIVE_PATH indicates the path for negative training samples, OUTPUT_PATH indicates the path for outputing transferred 
xls files into npy file path (for example: DriverBase/Orig_Data.npy). Set NEGATIVE_PATH to "None" to transfer a single xls file into npy format. LABEL_PATH
is for storing the label information of test data. If you are running DataLoader.py on training data, ignore this term. Note that this LABEL_PATH should be consistent
with the one in test.py's hyperparameters. Set TEST_LABEL_EXIST to "True" if you are running DataLoader.py on test data with labels provided. Otherwise leave it as default.
* outlier_detect.py is not compulsory, only adopted if removing outliers is useful. We use Isolation Forest to remove the outliers from the
data. INPUT_PATH is the path of DataLoader.py's OUTPUT_PATH, OUTPUT_PATH is for data with outliers removed (e.g. DriverBase/cleaned_data_orig.npy), 
and DATA_TYPE chosen from {"orig", "phred"}. Specific introduction can be found by using --help command. 
* train.py does the training process with 10-fold cross-validation, where the parameter space is defined in utils.py. DATA_PATH is the path for training data. 
METHOD can be chosen from SVM (svm), Gradient Boosting Tree (gbdt), Random Forest (rf),  Multi-layer Perceptron (mlp), Adaboost (adaboost) and XGBoost (xgbt).
* test.py does the testing. Model parameters need to be copied, models are saved during testing. If TRAIN is set to "True", then train and save models according to the best parameters, else no training is done. Thus
TRAIN should be set to "True" for the first time and the other times are optional. DATA_TYPE should take from {"orig", "phred"}, TEST_PATH is the path of test data. OUTPUT_FOLDER is a folder for saving prediction results, 
usually can make it the same directory as test data. LABEL_PATH is for loading the label information to generate output xls files. This path should be consistent with the one you indicated in DataLoader.py. Set TEST_LABEL_EXIST 
to "True" if you are running test.py on test data with labels provided. Otherwise leave it as default.
* analyze.py analyzes the data and trained model using Sklearn methods and SHAP analysis. Model parameters need to be determined in train.py and analyze.py should use these 
parameters for analysis (the best parameters for XGBoost are already available in code, but if you would like to change them, you have to do it manually). DATA_PATH indicates
the path for data. DATA_FORM takes a value from {"orig", "phred", "test"}. If you choose "test", the data provided should be only from test dataset.
  
### Examples
We provide some exmaples to show how to use the codes. 

* Loading training data matrix.
```python
python DataLoader.py -pp DriverBase/training_Y_orig.xls -pn DriverBase/training_N_orig.xls -op DriverBase/Orig_Data.npy 
python DataLoader.py -pp DriverBase/training_Y_Phred.xls -pn DriverBase/training_N_Phred.xls -op DriverBase/Phred_Data.npy
```
* Remove outliers from training data, run the following command if removing is useful (but anyway you need to run this to see if it works).
```python
python outlier_detect.py -ip DriverBase/Orig_Data.npy -op DriverBase/cleaned_data_orig.npy -t orig
python outlier_detect.py -ip DriverBase/Phred_Data.npy -op DriverBase/cleaned_data_phred.npy -t phred
```
* Perform training on the data using 10-fold cross-validation and determine best parameters using grid-search. We can employ XGBoost to build a cleaned Phred-scaled model using -m "xgbt" and -d "phred_cleaned".
```python
python train.py -p DriverBase/cleaned_data_phred.npy -m xgbt
```
* Testing model using independent data.
```python
python DataLoader.py -pp Test_Data_Final/Pancancer/Pancancer_positive_orig.xls -pn Test_Data_Final/Pancancer/Pancancer_negative_orig.xls -op Test_Data_Final/Pancancer/Phred_Data.npy -lp Test_Data_Final/Pancancer/label_phred.npy -l True
python test.py -f True -d phred -tp Test_Data_Final/Pancancer/Phred_Data.npy -of Test_Data_Final/Pancancer/ -lp Test_Data_Final/Pancancer/label_phred.npy -l True
```
* Evaluate the performance based on 10-fold cross-validation and independent test data. SHAP analysis of feature importance and SHAP value is also supported. You can change the source code to specify which kind of analysis to perform on. 
```python
python analyze.py -d phred -p DriverBase/cleaned_data_phred.npy
```
#### Notes for running code
- Be careful -of indicates a folder, not file. Recommended to add '/' at last
- XGBoost's parameters are determined two or three at a time but not all at the same time due to the CPU power. For faster training, you can also changhe the code to determine the parameters two or three at a time.


### Copyright
Copyright (c) @ University of Michigan 2019-2021. All rights reserved. Please note some constituent features of AI-Driver contain specific licence or usage constraints for non-academic usage. AI-Driver does not grant the non-academic usage of those scores, so please contact the original score/method providers for proper usage purpose.        
