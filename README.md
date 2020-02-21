# AI-Driver
AI-Driver is a ...

## Download
* Clone and fork code and from this repository
* Train data is available in the DriverBase folder, whole exome matrix data can be downloaded [Here](http://47.89.179.59/download/varcards.main.Phred_scaled.xls.gz)

## Usage
### Environment Requirement
You can install the environment by using conda. Please install [anaconda] (https://docs.anaconda.com/anaconda/install/) before install AI-Driver.
```python
conda create -n python3 python=3 xgboost=0.90 scikit-learn=0.221 xlrd xlwt pandas shap xlutils
```
where the used packages include
* Python 3.6
* Pandas (with xlrd>=0.9.0)
* Sklearn 0.22.1
* XGBoost 0.90
* xlrd, xlwt
* SHAP


### Basic Usage
* Run DataLoader.py on original raw Excel (.xls) files. This step extracts the useful information from files and shuffles the dataset for better robustness.
* Then run train.py by indicating the specific dataset and method. We use 10-fold cross-validation to determine the best parameters for each method. The datasets include original version, original+cleaned version, Phred version, Phred+cleaned version.The source of dataset can be changed according to your preference. Also, the range of parameters to be selected can be founded and changed in utils.py.  
* Analysis and testing of the models can be done in analyze.py and test.py.
* outlier_detect.py uses Isolation Forest to detect outliers in data and remove them from the dataset. The new datasets are "cleaned".
* The procedure can be described as the following code:
```python
python DataLoader.py -pp POSITIVE_PATH -pn NEGATIVE_PATH -op OUTPUT_PATH
python outlier_detect.py -ip INPUT_PATH -op OUTPUT_PATH -t DATA_TYPE
python train.py -d DATA_TYPE -m METHOD
python analyze.py -d DATA_FORM -p DATA_PATH
python test.py -f TRAIN -d DATA_TYPE -tp TEST_PATH -of OUTPUT_FOLDER
```
Explaination:
* Running DataLoader.py transforms the original xls files into npy file for continuous experiments. Missing value imputation 
and random shuffling of data is also done. POSITIVE_PATH indicates the path for positive training samples (for example: DriverBase/training_Y_orig.xls), 
NEGATIVE_PATH indicates the path for negative training samples, OUTPUT_PATH indicates the path for outputing transferred 
xls files into npy file path (for example: DriverBase/Orig_Data.npy). Set NEGATIVE_PATH to "None" to transfer a single xls file into npy format.
* outlier_detect.py is not compulsory, only adopted if removing outliers is useful. We use Isolation Forest to remove the outliers from the
data. INPUT_PATH is the path of DataLoader.py's OUTPUT_PATH, OUTPUT_PATH is for data with outliers removed (e.g. DriverBase/cleaned_data_orig.npy), 
and DATA_TYPE chosen from {"orig", "phred"}. Specific introduction can be found by using --help command. 
* train.py does the training process with 10-fold cross-validation, where the parameter space is defined in utils.py. DATA_TYPE take from {"orig", "phred", "orig_cleaned", "phred_cleaned"},  
which indicates original data, Phred data, original data with outliers removed and Phred data with outliers removed. METHOD can be chosen from SVM (svm), Gradient Boosting Tree (gbdt), Random Forest (rf), 
Multi-layer Perceptron (mlp), Adaboost (adaboost) and XGBoost (xgbt).
* analyze.py analyzes the data and trained model using Sklearn methods and SHAP analysis. Model parameters need to be determined in train.py and analyze.py should use these 
parameters for analysis (the best parameters for XGBoost are already available in code, but if you would like to change them, you have to do it manually). DATA_PATH indicates
the path for data. DATA_FORM takes a value from {"orig", "phred", "test"}. If you choose "test", the data provided should be only from test dataset.
* test.py does the testing. Model parameters need to be copied, models are saved during testing. If TRAIN is set to True, then train and save models according to the best parameters, else no training is done. Thus
TRAIN should be set to True for the first time and the other times are optional. DATA_TYPE should take from {"orig", "phred"}, TEST_PATH is the path of test data. OUTPUT_FOLDER is a folder for saving prediction results, 
usually can make it the same directory as test data. 
  

### Copyright
