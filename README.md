# Automatic Scoring of the Test des Deux-Barrages T2B

Paper-pencil tests are essential tools in neuropsychology to quantify effects of a disease on cognitive performance. The evaluation of these tests is mostly done manually, which is time-consuming and prone to errors. This thesis proposes an automatic scoring system for the Test des Deux Barrages, a selective attention test regularly used in Switzerland. The goal of the thesis was to investigate the number of errors made in the evaluation and to develop a scoring system that would deliver results similar to the manual evaluation by clinical experts. 
For this purpose, 23 filled-out test sheets and their evaluations by clinical experts were collected. The clinical evaluations were examined for errors using the ground truth of the patients. The collected test sheets were split into individual symbols and the symbols classified as marked by the patient or not based on their grayscale values. Three models were developed for binary classification. The performance of the models was evaluated on various performance metrics (balanced accuracy, F1 measure, sensitivity and specificity) and compared to the performance of clinical experts using the sign test and Wilcoxon signed-rank test.

# Overview:

In this repository you can find:
- the data used and generated in this thesis (see 'Data')
- the script used for performance analysis (see 'Performance Analysis')
- the script used for statistical tests (see 'Statistical Tests')
- the script used to plot figures (see 'Plots') + link to Google Colab Notebook for remainder of figures
- a script to convert .pdf files to images ('tkpdf2img.py')
- the Master thesis
- GUI and streamlit application using an updated version (automatic rotation and skew correction) of the script (see 'Applications')

# How to navigate:

Folders were specifically created: Scripts within a folder use the files within the folder. Please use the files that were included in the same folder to ensure that the scripts run correctly. Tkinter windows will pop up when running the code, prompting to select files. The window titles will specify which files to choose. 
As PyCharm does not support seperate code blocks (to the best of my knowledge) like Google Colab or Visual Studio Code, the main script was seperated into individual scripts. All individual scripts should run independently, with all necessary functions included within the scripts.

For performance analysis.py:
- To run the performance analysis of the clinicians use the files in folder 'Performance Analysis - Clinicians'. Both .xlsx (clinician's evaluation) and .xlsm (ground truth of patient's) Excel files need to be in the same folder. The code is designed to access both types of Excel files by name. The tkinter window will prompt you to select the .xlsx files and the script will automatically select the corresponding .xlsm files as long as they are in the same folder and have the same name (T2B_date_...). At the end of the code a new Excel file will be generated with the calculated performance analysis of the clinicians.

- To run the performance analysis of the models a similar rule applies. Use the files in folder 'Performance analysis - Script', consisting of .xlsm files (ground truth of patient's) and test sheet images (.jpeg). The tkinter window will prompt you to select the ground truth files and the script will automatically use the corresponding images to run the classification models on, as long as they are in the same folder and have the same name (T2B_date_...). At the end of the code a new Excel file will be generated with the calculated performance analysis of the models.

For statistical tests.py:
- To run the sign test and Wilcoxon signed-rank test use the files in folder 'Statistical tests'. These Excel files contain the performance analysis of the clinicians and the models. The tkinter window title will specify which file to select, when the tkinter window prompts you to select a file. After selecting the correct files the script will run the statistical tests and print the results. 
