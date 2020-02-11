## DiverBase
A interdisciplinary project (Gnome+ML). Details would be discussed later. 

##Download
* Code can be cloned and forked from this repository
* Data can be downloaded from ...

##Usage
###Environment Requirement
* Python 2.7 (/3.6) (does not matter much)
* Pandas (with xlrd>=0.9.0)
* Sklearn 0.22.1
* XGBoost 0.90

###Basic Usage
* Run DataLoader.py on original raw Excel (.xls) files. This step extracts the useful information from files and shuffles the dataset for better robustness.
* Then run train.py by indicating the specific dataset and method. We use 10-fold cross-validation to determine the best parameters for each method. The datasets include original version, original+cleaned version, Phred version, Phred+cleaned version.The source of dataset can be changed according to your preference. Also, the range of parameters to be selected can be founded and changed in utils.py.  
* Analysis and testing of the models can be done in analyze.py and test.py.
* outlier_detect.py uses Isolation Forest to detect outliers in data and remove them from the dataset. The new datasets are "cleaned".
* The procedure can be described as the following code 
```python
python DataLoader.py
python outlier_detect.py
python train.py -d phred -m xgbt
python analyze.py
python test.py
```

##Experiment Details 

#### Data Preprocessing
* We train and test on both original data scores, as well as PHRED-scaled scores
* The data are shuffled to obtain better robustness for the results
* Original data features are selected through ROC analysis and SHAP feature evaluation 

### Methods Used for Experiments
We adopt some well-known ensemble methods:
* Gradient Boosting Decision Tree (GBDT)
* SVM 
* AdaBoost 
* Random Forest (RF)
* Multi-layer Perceptron (MLP)
* XGBoost (xgbt)

