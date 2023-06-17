# REPRODUCIBLE RESEARCH PROJECT DOCUMENTATION

### AUTHORS: RAFAŁ MISIÓRSKI, MICHAŁ STRYJEK, KAMIL GOLIS

## 1. PROJECT OVERVIEW

### Introduction

The aim of our project is to reproduce the project originally written in R language (in a form of Rmd file - ``) in Python (using Jupyter notebooks) and then extend the original analysis. 

The original project was created as an ad-hoc analysis for the sake of other course at University of Warsaw and we found much space to employ and implement reproducible research prinnciples here.

We divided our reproduction process into two parts:
1. We translated the solution from R to Python as closely as possible in a fully reproducible manner.
2. We simplified and extended the analysis, employing good coding practices and using externally defined functions to avoid mistakes present in the original work and extend the analysis.


### Reproducibility 

The original R solution is not fully reproducible. In order to catch one version of reality to be then compared with reproduced Python results, we rendered the Rmd script to html format and listed versions of R and libraries used in the analysis (in the file `software_version.txt`).

The whole Python solution (parts II and III) is fully reproducible. In order to achieve the same results, one should install python environment with packages (in appropriate version) used through the analysis according to file `requirements.txt` (present in the main folder of the repository). We used the Windows 10 operating system and python in version 3.8. In order to create the environment, simply type in the following command:

conda create -n <env_name> python=3.8 --file <path_to_requirements>\requirements.txt

Then activate the environment:

conda activate <env_name>

In the case or problem while collecting the required libraries (we came across such a problem), before the environment creation, use the following command:

conda config --add channels conda-forge

Once the environment is ready and set up - one can reproduce our analysis.


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

TBA


## 3. R-TO-PYTHON TRANSLATION AND REPRODUCTION (II_reproduced_original_Python)

TBA


## 4. EXTENDED ANALYSIS IN PYTHON - IMPLEMENTING GOOD CODING/REPRODUCING PRACTICES (III_best_practices_Python)

TBA


## 5. SUMMARY

TBA


## 6. FILES STRUCTURE

TBA