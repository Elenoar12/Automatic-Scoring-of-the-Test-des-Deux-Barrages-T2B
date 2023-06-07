import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from scipy.stats import wilcoxon
from scipy.stats import binomtest

root = tk.Tk()
root.withdraw()                                                                                                         #use to hide tkinter window

### Statistical tests:
# To run the sign test and Wilcoxon signed-rank test use the files in folder 'Statistical tests'. These Excel files
# contain the performance analysis of the clinicians and the models. The tkinter window title will specify which file to
# select, when the tkinter window prompts you to select a file. After selecting the correct files the script will run
# the statistical tests and print the results.

## choose performance analysis files:

def search_for_clin_perf_path():
    excel_paths = filedialog.askopenfilename(parent = root, title = 'Select performance analysis of clinicians', filetypes = (("Excel files","*.xlsx"),("all files","*.*")))
    return excel_paths

def search_for_model_in_or_out_perf_path():
    excel_paths = filedialog.askopenfilename(parent = root, title = 'Select performance analysis of Model with inside OR outside criteria', filetypes = (("Excel files","*.xlsx"),("all files","*.*")))
    return excel_paths

def search_for_model_in_and_out_perf_path():
    excel_paths = filedialog.askopenfilename(parent = root, title = 'Select performance analysis of Model with inside AND outside criteria', filetypes = (("Excel files","*.xlsx"),("all files","*.*")))
    return excel_paths

def search_for_model_strict_threshold_perf_path():
    excel_paths = filedialog.askopenfilename(parent = root, title = 'Select performance analysis of Model with strict threshold for non-target symbols', filetypes = (("Excel files","*.xlsx"),("all files","*.*")))
    return excel_paths

## choose performance analysis and prepare pd dataframe for comparison:

# choose performance analysis of clinicians:
clin_perf_path = search_for_clin_perf_path()
clin_perf = pd.read_excel(clin_perf_path)
del clin_perf[clin_perf.columns[0]]
clin_perf = clin_perf.iloc[:-2 , -4:]
clin_perf.to_numpy

# choose performance analysis of Model with inside OR outside criteria:
model_in_or_out_perf_path = search_for_model_in_or_out_perf_path()
Model_in_or_out_perf = pd.read_excel(model_in_or_out_perf_path)
del Model_in_or_out_perf[Model_in_or_out_perf.columns[0]]
Model_in_or_out_perf = Model_in_or_out_perf.iloc[:-2 , -4:]
Model_in_or_out_perf.to_numpy

# choose performance analysis of Model with inside AND outside criteria:
model_in_and_out_perf_path = search_for_model_in_and_out_perf_path()
Model_in_and_out_perf = pd.read_excel(model_in_and_out_perf_path)
del Model_in_and_out_perf[Model_in_and_out_perf.columns[0]]
Model_in_and_out_perf = Model_in_and_out_perf.iloc[:-2 , -4:]
Model_in_and_out_perf.to_numpy

# choose performance analysis of Model with strict threshold for non-target symbols:
model_strict_threshold_perf_path = search_for_model_strict_threshold_perf_path()
Model_strict_threshold_perf = pd.read_excel(model_strict_threshold_perf_path)
del Model_strict_threshold_perf[Model_strict_threshold_perf.columns[0]]
Model_strict_threshold_perf = Model_strict_threshold_perf.iloc[:-2 , -4:]
Model_strict_threshold_perf.to_numpy

## Sign test - Clinician vs. Model in_or_out criteria:
diff = clin_perf.subtract(Model_in_or_out_perf)

pos_diff = np.sum(diff['b-ACC'].to_numpy() > 0)
neg_diff = np.sum(diff['b-ACC'].to_numpy() < 0)
zero_diff = np.sum(diff['b-ACC'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model in_OR_out (b-ACC):', result)

pos_diff = np.sum(diff['F1'].to_numpy() > 0)
neg_diff = np.sum(diff['F1'].to_numpy() < 0)
zero_diff = np.sum(diff['F1'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model in_OR_out (F1):', result)

pos_diff = np.sum(diff['SEN'].to_numpy() > 0)
neg_diff = np.sum(diff['SEN'].to_numpy() < 0)
zero_diff = np.sum(diff['SEN'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model in_OR_out (SEN):', result)

pos_diff = np.sum(diff['SPE'].to_numpy() > 0)
neg_diff = np.sum(diff['SPE'].to_numpy() < 0)
zero_diff = np.sum(diff['SPE'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model in_OR_out (SPE):', result)

## Wilcoxon signed-rank test - Clinician vs. Model in_or_out criteria:
results_Model_in_or_out = wilcoxon(clin_perf, Model_in_or_out_perf, zero_method='zsplit')
print('Wilcoxon results - Model in_OR_out [b-ACC, F1, SEN, SPE]:', results_Model_in_or_out)

## Sign test - Clinician vs. Model in_and_out criteria:
diff = clin_perf.subtract(Model_in_and_out_perf)

pos_diff = np.sum(diff['b-ACC'].to_numpy() > 0)
neg_diff = np.sum(diff['b-ACC'].to_numpy() < 0)
zero_diff = np.sum(diff['b-ACC'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model in_AND_out (b-ACC):', result)

pos_diff = np.sum(diff['F1'].to_numpy() > 0)
neg_diff = np.sum(diff['F1'].to_numpy() < 0)
zero_diff = np.sum(diff['F1'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model in_AND_out (F1):', result)

pos_diff = np.sum(diff['SEN'].to_numpy() > 0)
neg_diff = np.sum(diff['SEN'].to_numpy() < 0)
zero_diff = np.sum(diff['SEN'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model in_AND_out (SEN):', result)

pos_diff = np.sum(diff['SPE'].to_numpy() > 0)
neg_diff = np.sum(diff['SPE'].to_numpy() < 0)
zero_diff = np.sum(diff['SPE'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model in_AND_out (SPE):', result)

## Wilcoxon signed-rank test - Clinician vs. Model in_and_out criteria:
results_Model_in_and_out = wilcoxon(clin_perf, Model_in_and_out_perf, zero_method='zsplit')
print('Wilcoxon results - Model in_AND_out [b-ACC, F1, SEN, SPE]:', results_Model_in_and_out)

## Sign test - Clinician vs. Model with strict threshold for non-target symbols:
diff = clin_perf.subtract(Model_strict_threshold_perf)

pos_diff = np.sum(diff['b-ACC'].to_numpy() > 0)
neg_diff = np.sum(diff['b-ACC'].to_numpy() < 0)
zero_diff = np.sum(diff['b-ACC'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model_strict_threshold (b-ACC):', result)

pos_diff = np.sum(diff['F1'].to_numpy() > 0)
neg_diff = np.sum(diff['F1'].to_numpy() < 0)
zero_diff = np.sum(diff['F1'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model_strict_threshold (F1):', result)

pos_diff = np.sum(diff['SEN'].to_numpy() > 0)
neg_diff = np.sum(diff['SEN'].to_numpy() < 0)
zero_diff = np.sum(diff['SEN'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model_strict_threshold (SEN):', result)

pos_diff = np.sum(diff['SPE'].to_numpy() > 0)
neg_diff = np.sum(diff['SPE'].to_numpy() < 0)
zero_diff = np.sum(diff['SPE'].to_numpy() == 0)
tot_obsv = len(diff) - zero_diff
result = binomtest(min(pos_diff, neg_diff), n=tot_obsv, p=0.5)
print('Sign test results - Model_strict_threshold (SPE):', result)

## Wilcoxon signed-rank test - Clinician vs. Model with strict threshold for non-target symbols:
results_Model_strict_threshold = wilcoxon(clin_perf, Model_strict_threshold_perf, zero_method='zsplit')
print('Wilcoxon results - Model_strict_threshold [b-ACC, F1, SEN, SPE]:', results_Model_strict_threshold)
