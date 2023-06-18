# REPRODUCIBLE RESEARCH PROJECT DOCUMENTATION

### AUTHORS: RAFAŁ MISIÓRSKI, MICHAŁ STRYJEK, KAMIL GOLIS

## 1. PROJECT OVERVIEW

### Introduction

The aim of our project is to reproduce the project originally written in R language (in a form of Rmd file - `Energy_prediction_original.Rmd`) in Python (using Jupyter notebooks) and then extend the original analysis. 

The original project was created as an ad-hoc analysis for the sake of other course at University of Warsaw and we found much space to employ and implement reproducible research prinnciples here.

We divided our reproduction process into two parts:
1. We translated the solution from R to Python as closely as possible in a fully reproducible manner.
2. We simplified and extended the analysis, employing good coding practices and using externally defined functions to avoid mistakes present in the original work and extend the analysis.


### Reproducibility 

The original R solution is not fully reproducible. In order to catch one version of reality to be then compared with reproduced Python results, we rendered the Rmd script to html format and listed versions of R and libraries used in the analysis (in the file `software_version.txt`).

The whole Python solution (parts II and III) is fully reproducible. In order to achieve the same results, one should install python environment with packages (in appropriate version) used through the analysis according to file `requirements.txt` (present in the main folder of the repository). We used the Windows 10 operating system, Python in version 3.8 and Anaconda platform. 

REPRODUCTION STEPS:

1. Install required software (Windows 10, python 3.8, Anaconda)
2. From Windows run Anaconda Prompt
3. run command: conda create -n <env_name> python=3.8 --file <path_to_requirements>\requirements.txt
   
    <env_name> can be selected freely and refers to the environment name of the new virtual environment
   
    <path_to_requirements> is the path to the project folder in the system. It can be skipped if the Anaconda Prompt is run from the project directory

In the case or problem while collecting the required libraries (we came across such a problem), before the environment creation, use the following command:

conda config --add channels conda-forge

4. Activate the environment. Run command: conda activate <env_name>
5. Move to III_best_practices_Python (command: dir <path_to_project>/III_best_practices_Python)
6. Run Jupyter Notebook from this folder (command: jupyter notebook)
7. From within the jupyter notebook UI open the Energy_prediction_best_practices_extended_Python.ipynb
8. Run the code


### Sources Declaration

Throughout the reproduction process we used a few different sources to obtain functions/solutions for our problems. The exact sources are indicated in the jupyter notebooks with the reproduced solution, but we list them also here, to make it clear, where and how we used external information provided by other people:

1. We needed to set several seeds at the start of the notebooks in order to provide full reproducibility. We followed the instructions outlined here: https://keras.io/getting_started/faq/ (section How can I obtain reproducible results using Keras during development?) and here: https://machinelearningmastery.com/reproducible-results-neural-networks-keras/ (section Seed Random Numbers with the TensorFlow Backend).
2. The function to calculate VIF - we needed a function to calculate Variance Inflation Factor in Python as closely as in R. For this purpose we used (and slightly modified) a function provided in the StackOverflow thread concerning VIF. The exact source: https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python Answer provided by user `steven`, on Feb 24, 2019 at 23:06.
3. We created a function to conduct an automatic feature selection based on VIF using the abovementioned VIF calculation function. Within the defined automatic selection function, we also used method outlined here: https://note.nkmk.me/en/python-dict-get-key-from-value/, the exact source: http://localhost:8923/lab/tree/rr_project_work/structure/III_best_practices_Python/src/rr_IV_select_features.ipynb, function: `get_key_from_value(d, val)`.
4. In order to plot the loss and MAPE for training history of CNN in a similar way as in the original solution, we created plots based on the following article: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/ (section Visualize Model Training History in Keras, lines 22-37). 
5. Finally, we provide once more the source of the data used in the analysis: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction


### Reproduction Process

This repository and the documentation is divided into three major parts reflecting the order of the analysis: 
* the original solution (directory `I_original_solution_R`)
* R-to-Python translation and reproduction (directory `II_reproduced_original_Python`)
* Extended analysis in Python (directory `III_best_practices_Python`)

We are going to describe each part of the analysis and reproduction process and then summarise the whole reproduction process.



## 2. ORIGINAL SOLUTION (I_original_solution_R)

All of the information in this section pertains to files in the directory `I_original_solution_R`.


The original solution is the Rmd file contatining the analysis for regression problem. At the start, there were a few files with the (The whole project (regression+classification) `Machine_Learning_2_Project_RM.Rmd`, regression problem solution using XGBoost `XG_Boost_approach_3_w_standardization.R` and regression problem solution using CNN `CNN_approach.R`). Since our aim is to reproduce the regression problem, and due to fact that the separate analyses for XGBoost and CNN are also included in the main Rmd file, we decided to modify the `Machine_Learning_2_Project_RM.Rmd` in a way to include only the regression problem, and we moved the rest of the files to directory `original_files_irrelevant`. The final file with original solution is named as `Energy_prediction_original.Rmd`.


In the original Rmd file, the task is to predict the appliancies energy use in low-energy building in 10-minute intervals. The autor used variance inflation factor analysis in order to select features used in the modelling process (in a way to avoid multicollinearity). The author trained XGBoost and Convolutional Neural Network models and applied them for predictions. The evaluation metric for the results was mean absolute percentage error.


Throughout the translation to Python and reproduction process, we found many inconsistencies and challenges, that we have to tackle. These challenges were:


* The R solution was not fully reproducible. As stated earlier, in order to provide the reproducibility at least to some degree, we rendered the Rmd script to html format and listed versions of R and libraries used in the analysis.
* There was no clear structure stated in the original file. Based on the content of the Rmd file, we outlined the five major parts of the analysis: 1. Import libraries used in the analysis, 2. Exploratory Data Analysis, 3. Feature Engineering, 4. Feature Selection and  5. Modelling.
* The data was processed multiple times, the resulting datasets were overwritten in the next steps, which ended with choosing wrong variable as target for CNN. We reproduced these mistakes in part II, and then dealt with them in part III. To do so, we wrote external functions to prepare and process the data once - then the exact same dataset could be used to build various ML models. In part III we also fixed the target name at the beginning of the analysis to avoid the problem of assigning the wrong variable.
* The feature selection process using VIF calculation was unclear and messy in the original work - the excluded features were not always the ones having the highest VIF. Most probably it was due to repeated manual calculation and declaration which variable is to be excluded. We reproduce this approach in Python implementation in part II, and then deal with this problem by implementing a function to select the predictors automatically.
* The dataset was divided into train/test in different ways for XGBoost and CNN, and thus the results obtained for these models are not comparable. In part II we reproduce this approach, but in part III we employ a consistent method to assign the observations to train and test datasets (we do it once through the analysis).
* The results obtained for CNN in original work were different from the ones declared by author in the Rmd file. He mistook the values obtained for the CNN training and prediction. We re-ran and rendered the original Rmd file to provide a basis for comparison with solution reproduced in Python.



## 3. R-TO-PYTHON TRANSLATION AND REPRODUCTION (II_reproduced_original_Python)

All of the information in this section pertains to files in the directory `II_reproduced_original_Python`.


The aim of this part was to translate from R to Python the original solution, and reproduce the results **as closely as possible**. The analysis is fully reproducible thanks to declaring the software version and setting several seeds. We adopted the approach outlined in the original solution, including the inconsistencies present in the original solution. The analysis consists of five parts, as stated earlier (according to the content of original file):


1. Import libraries used in the analysis
2. Exploratory Data Analysis
3. Feature Engineering
4. Feature Selection
5. Modelling


In part 1 we import the libraries needed for the analysis. We expected, that the reproduced solution should be quite similar to the original one, as in original R solution author used  `keras` and `tensorflow` (installed through `reticulate`), and the `XGBoost` library is available both in Python and R - we could use the same architecture/hyperparameters, which increased our chance to succeed.


In part 2 we conducted an EDA. All of the functions used in R solution had their close reference in Python (e.g. R `summary` vs. Python `describe`). It was pretty simple to reproduce the results here.


In part 3 we created additional features which are used later in the analysis. The feature engineering pertains to the `date` column (in fact this variable contained information about date and time of the observations made). In Python solution we used for this task `datetime` library and the results are exactly the same as in the case of original Rmd file. We noticed, that some of the created features are then excluded (since they were created as character, not numeric) - here we recreate it, but in part III we use an external function for the feature engineering and we recode all features as numeric type.


In part 4 we recreated the feature selection mechanism present in the original file - by calculating variance inflation factor values for the regressors. This part was a bit challenging, as there was no straightforward way to calculate VIF in Python to obtain results similar to those obtained in R (using the `car` library). To overcome this challenge, we used and modified slightly the function from StackOverflow thread (exact source avaialble in the `Sources Declaration` section). Thanks to the function, we obtained almost the same results as in the R original solution. We followed the method proposed in original solution and calculated and excluded the variables manually - we automate the feature selection process in part III, where we introduce an external function to do it.


In part 5 we prepared the models - XGBoost and CNN. We expected that the results should be similar, yet there were two problems: a minor one and a major one. The minor one was the fact, that the original Rmd solution was not fully reproducible (The results of CNN were difefrent each time we run the code). The second one was bigger, as the author mistook the MAPE values and stated that MAPE for training set was 5.4%, while for the test set 4.74%. We made multiple re-runs of the file and rendered it - the resulting MAPE values were set around 110-115%. We prepared the Python version of the XGBoost and CNN models and the results were very similar:


* XGBoost - test data MAPE: R - 24.4%, Python - 23.8% - success!
* CNN - train/test datea MAPE: R - ~110%/~115%, Python - 112.3%/110.1% - there are some differences, but we perceive the results reproduction as successful!


We spotted the mistakes and inconsistencies in the original work, which were also present in the reproduced version translated to Python. However, it is not the end of our analysis. We create both simplified and extended analysis assuming good coding practices in the next part.



## 4. EXTENDED ANALYSIS IN PYTHON - IMPLEMENTING GOOD CODING/REPRODUCING PRACTICES (III_best_practices_Python)

All of the information in this section pertains to files in the directory III_best_practices_Python.

In this section we built several external functions and used them to prepare the analysis in proper way. The results obtained in this version are:

* simplified - we used external functions to minimise the code present in the main jupyter notebook and to provide a clear process to the analysis (prepare the data once).
* extended - the prepared functions enable the user to built various models and assume various evaluation metrics for the sake of analysis - everything available by calling one function.

What is important - the results for this part are fully reproducible, but they do not reproduce the original findings - in this part we correct the mistakes spotted in the original version, which affects the results.

This part consists of two main parts:

* external functions, which are the building blocks (present in the subdirectory src). They are covered in four Jupyter Notebooks, which are then loaded in the main one (we used jupyter notebooks instead of python scripts in order to simplify the dependencies of the external functions on other libraries). They are named in a way to reflect the structure of the analysis (note - there is no rr_II file, as the EDA consisted of a set of easy to use functions - we still do it manually):
    * rr_I_load_packages.ipynb - refers to the part I - Import libraries - here we simply import required libraries.
    * rr_III_prepare_data.ipynb - refers to the part III - Feature Engineering - here we define two functions: one for reading the csv file to pandas DataFrame, and the second one for feature engineering (exact functions description available inside the file).
    * rr_IV_select_features.ipynb - refers to the part IV - Feature Selection. We define it here a function to select features according to VIF values automatically (exact function description available inside the file).
    * rr_V_prepare_evaluate_model.ipynb - refers to the part 5 - Modelling - we define here two functions. The first one prepares the objects ready to create model (standard X/y train/test, assuming a defined train/test split). The second one fits a chosen model with defined hyperparameters and evaluates the predictions on train and test data using a chosen evaluation metric. Exact functions description available inside the file
* The main jupyter file, where we apply the external functions (Energy_prediction_best_practices_extended_Python.ipynb - we also rendered it as html file - Energy_prediction_best_practices_extended_Python.html)

In the file Energy_prediction_best_practices_extended_Python.ipynb we firstly go through the analysis in the order outlined earlier (through points 1-5, libraries imports, EDA, feature engineering and selection and modelling). The process is straightforward and far less code-consuming, as we use our external functions. Then, as a bonus, we provide an additional analysis. We build a set of various models built by calling one function (we prepared XGBoost, Linear Regression, Light GBM and Random Forest models, but sky is the limit - the function is very universal). We also applied various evaluation metrics (MAPE/MAE/R^2/MSE). Finally, we built different models using the same evaluation metric to make them comparable (in comparison with the models created in original work). 



## 5. SUMMARY

The reproduction process was challenging, yet we managed to reproduce the results from original R solution to Python and then simplified and extended the analysis by assuming good practices learned through the course. The external functions defined in part III could be successfully used in further analysis or for the sake of analysis for other time-series data.  

Our work is fully reproducible, what we appreciated as a result of the code translation. The source solution lacked the full reproducibility (XGBoost results reproducible, but CNN not), the clarity and consistency in terms of the data operations made to produce the models. Moreover, the original solution used a feature selection mechanism, which was iterated manytimes manually, which favours making mistakes in terms of rxcluding inappropriate variable. Finally, the inconsistencies in the code led to assuming wrong target variable for the CNN model training.

We managed to recreate the whole process in Python, and then we corrected these mistakes in part III - we refer to it as the dream original version of the analysis.


## 6. FILES STRUCTURE

TBA
