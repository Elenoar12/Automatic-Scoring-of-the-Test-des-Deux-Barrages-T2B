import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import statsmodels.api as sm
from scipy.signal import argrelmax, argrelmin
from scipy.sparse import csr_matrix
import pandas as pd
import os
from scipy import ndimage


root = tk.Tk()
root.withdraw()                                                                                                         #use to hide tkinter window

def search_for_xlsm_paths():
    excel_paths = filedialog.askopenfilenames(parent = root, title = 'Select ground truth files', filetypes = (("Excel files","*.xlsm"),("all files","*.*")))
    return excel_paths

def search_for_xlsx_paths():
    excel_paths = filedialog.askopenfilenames(parent = root, title = 'Select clinicians evaluation files (.xlsx)', filetypes = (("Excel files","*.xlsx"),("all files","*.*")))
    return excel_paths

def T2B_import(jpeg_path):
  img = cv.imread(jpeg_path, cv.IMREAD_GRAYSCALE)                                                                       #if test sheets to be rotated --> img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE) or COUNTERCLOCKWISE
  dim = img.shape
  return img, dim

def T2B_splitter(img, dim, frac_row = 0.027, frac_clmn = 0.04):
  mean_row = np.mean(img, axis=1)                                                                                       #axis = 1 means working along the row
  mean_clmn = np.mean(img, axis=0)                                                                                      #axis = 0 means along the column
  x = np.array(range(dim[0]))
  y = np.array(range(dim[-1]))
  #smoothing:
  lowess = sm.nonparametric.lowess
  smth_row = lowess(mean_row, range(len(x)), return_sorted = False,
                    frac = frac_row)                                                                                    #frac parameter for clmn/row are opposite because x/y are swapped when rotated 90 degree
  row_peaks = argrelmax(smth_row)

  smth_clmn = lowess(mean_clmn, range(len(y)), return_sorted = False,
                     frac = frac_clmn)
  clmn_peaks = argrelmax(smth_clmn)
  #splits:
  row_peaks = np.asarray(row_peaks)                                                                                     #original row and column vals saved as tuple[ndarray]
  clmn_peaks = np.asarray(clmn_peaks)
  row_peaks = row_peaks.flatten(order='C')
  clmn_peaks = clmn_peaks.flatten(order='C')
  return row_peaks, clmn_peaks

def row_adaptor(row_peaks, exp_range = (300, 3050)):
  for std in range(3, 6):                                                                                               #std = 3, 4, 5 work the best, is there alternative to np.floor(np.std(np.diff(x)))??
    distance = np.diff(row_peaks)
    median = int(np.median(np.diff(row_peaks)))
    up_bound = median + std
    low_bound = median - std
    row_peaks = row_peaks[np.where(np.logical_and(row_peaks >= exp_range[0],
                                                    row_peaks <= exp_range[1]))]                                         #filters clmn_peak for expected range
    too_close = np.array(np.where(np.logical_or(np.diff(row_peaks) < low_bound,
                                      np.diff(row_peaks) > up_bound))).flatten()
    dis_too_close = np.diff(too_close).flatten()
    indx_too_close = (np.array(np.where(dis_too_close == 1)).
                      flatten() + 1).astype(int)                                                                        #+1 returns indices of indices of peaks that are too close to each other
    indx_too_close = too_close[indx_too_close]
    frame = np.delete(row_peaks, indx_too_close)
    split = np.insert(frame, 0, (frame[0] - median))
    split = np.append(split, (frame[-1] + median))
    while len(split) == 41:
      return split

def clmn_adaptor(clmn_peaks, exp_range = (300, 2000)):
  for std in range(3, 6):                                                                                               #std = 3, 4, 5 work the best, is there alternative to np.floor(np.std(np.diff(x)))??
    distance = np.diff(clmn_peaks)
    median = int(np.median(np.diff(clmn_peaks)))
    up_bound = median + std
    low_bound = median - std
    clmn_peaks = clmn_peaks[np.where(np.logical_and(clmn_peaks >= exp_range[0],
                                                    clmn_peaks <= exp_range[1]))]                                       #filters clmn_peak for expected range
    too_close = np.array(np.where(np.logical_or(np.diff(clmn_peaks) < low_bound,
                                      np.diff(clmn_peaks) > up_bound))).flatten()
    dis_too_close = np.diff(too_close).flatten()
    indx_too_close = (np.array(np.where(dis_too_close == 1)).
                      flatten() + 1).astype(int)                                                                        #+1 returns indices of indices of peaks that are too close to each other
    indx_too_close = too_close[indx_too_close]
    frame = np.delete(clmn_peaks, indx_too_close)
    split = np.insert(frame, 0, (frame[0] - median))
    split = np.append(split, (frame[-1] + median))
    while len(split) == 26:
      return split

def sym_list_loop(img, adapt_row, adapt_clmn):                                                                          #loop for single symbols:
  row_range = np.array(range(len(adapt_row)))
  clmn_range = np.array(range(len(adapt_clmn)))
  sym_list = []
  mean_sym_list = []
  for i in row_range[1:]:                                                                                               #von index 1 ansonsten ist i-1 oder j-1 out of bounds
    for j in clmn_range[1:]:
      symbols = img[adapt_row[i-1]:adapt_row[i], adapt_clmn[j-1]:adapt_clmn[j]]
      if i == len(row_range):
        break
        if j == len(clmn_range):
          break
      sym_list.append(symbols)
      mean_sym_list.append(np.mean(symbols))
  return sym_list, mean_sym_list

def symbol_splitter(x, val = 210, frac_sym = 0.11):                                                                     #function to find row and clmn vals of single symbols:
  dim_sym = x.shape
  sym_mean_row = np.mean(x, axis=1)
  sym_mean_clmn = np.mean(x, axis=0)
  row_x = np.array(range(dim_sym[0]))
  clmn_y = np.array(range(dim_sym[-1]))
  lowess = sm.nonparametric.lowess
  sym_smth_row = lowess(sym_mean_row, range(len(row_x)),
                        return_sorted = False, frac = frac_sym)                                                         #adjusted from 0.15 to 0.11 because some symbols had no vals!
  sym_row_vals = np.asarray(argrelmin(sym_smth_row)).flatten()
  sym_vals = sym_smth_row[sym_row_vals]
  sym_vals_indx = np.asarray(np.where(sym_vals < val)).flatten()
  row_vals = sym_row_vals[sym_vals_indx]
  if len(row_vals) == 1:                                                                                                #symbols with only one minima
    if row_vals[0] < 30:
      row_vals = np.append(row_vals, (row_vals[0] + 21))                                                                #symbol number 412 (row 17/ clmn 13) and 712 (row 29/ clmn 13) don't have second minima!!!!!
    elif row_vals[0] > 30:
      row_vals = np.insert(row_vals, 0, (row_vals[0] - 21))                                                             #symbol number 690 (row 28/ clmn 16) doesn't have first minima!!!!!
  if len(row_vals) != 2:                                                                                                #symbols having more than two minima
    row_vals = row_vals[0], row_vals[-1]
  if np.diff(row_vals) < 15:
    if row_vals[0] > 30:
      row_vals = np.insert(row_vals, 0, (row_vals[1] - 21))
    elif row_vals[1] < 40:                                                                                              #symbols missing one minima, but having two minima
      row_vals = np.append(row_vals, (row_vals[0] + 21))
    row_vals = np.asarray([row_vals[0], row_vals[-1]])
  sym_smth_clmn = lowess(sym_mean_clmn, range(len(clmn_y)),
                         return_sorted = False, frac = frac_sym)
  sym_clmn_vals = np.asarray(argrelmin(sym_smth_clmn)).flatten()
  sym_vals = sym_smth_clmn[sym_clmn_vals]
  sym_vals_indx = np.asarray(np.where(sym_vals < val)).flatten()
  clmn_vals = sym_clmn_vals[sym_vals_indx]
  if len(clmn_vals) == 1:                                                                                               #symbols with only one minima
    if clmn_vals[0] < 30:
      clmn_vals = np.append(clmn_vals, (clmn_vals[0] + 21))                                                             #symbol number 412 (row 17/ clmn 13) and 712 (row 29/ clmn 13) don't have second minima!!!!!
    elif clmn_vals[0] > 30:
      clmn_vals = np.insert(clmn_vals, 0, (clmn_vals[0] - 21))                                                          #symbol number 920 (row 37/ clmn 13) doesn't have first minima!!!!!
  if len(clmn_vals) != 2:                                                                                               #symbols having more than two minima
    clmn_vals = clmn_vals[0], clmn_vals[-1]
  if np.diff(clmn_vals) < 15:                                                                                           #symbols missing one minima, but having two minima
    if clmn_vals[0] > 30:
      clmn_vals = np.insert(clmn_vals, 0, (clmn_vals[1] - 21))
    elif clmn_vals[1] < 40:
      clmn_vals = np.append(clmn_vals, (clmn_vals[0] + 21))
    clmn_vals = np.asarray([clmn_vals[0], clmn_vals[-1]])
  return dim_sym, row_vals, clmn_vals

def model_in_OR_out(sym_list, mean_sym_list, thresh = 250, corr = False):
  #symbol variations generated from empty test sheet:
  sym_var = [4, 3, 5, 0, 4, 7, 6, 1, 1, 7, 3, 5, 0, 4, 3, 7, 1, 7, 1, 2, 4, 3,
       5, 0, 4, 1, 7, 7, 1, 1, 4, 3, 5, 0, 4, 5, 0, 4, 3, 5, 6, 2, 6, 6,
       7, 1, 2, 6, 6, 1, 3, 5, 0, 4, 3, 1, 7, 6, 2, 1, 7, 1, 1, 7, 6, 0,
       4, 3, 5, 0, 3, 5, 0, 4, 3, 7, 7, 2, 7, 7, 4, 3, 5, 0, 4, 1, 6, 2,
       1, 6, 3, 5, 0, 4, 3, 7, 6, 2, 7, 7, 0, 4, 3, 5, 0, 2, 2, 6, 7, 2,
       0, 4, 3, 5, 0, 7, 6, 6, 7, 6, 0, 4, 3, 5, 0, 4, 3, 5, 0, 4, 1, 1,
       2, 7, 1, 5, 0, 4, 3, 5, 2, 7, 2, 6, 7, 4, 3, 5, 0, 4, 2, 7, 2, 6,
       2, 3, 5, 0, 4, 3, 6, 7, 1, 6, 6, 4, 3, 5, 0, 4, 2, 6, 2, 2, 2, 5,
       0, 4, 3, 5, 6, 7, 2, 1, 6, 2, 1, 7, 2, 7, 5, 0, 4, 3, 5, 5, 0, 4,
       3, 5, 2, 2, 7, 7, 6, 0, 4, 3, 5, 0, 0, 4, 3, 5, 0, 2, 1, 1, 6, 2,
       2, 6, 1, 1, 1, 3, 5, 0, 4, 3, 2, 1, 1, 7, 7, 6, 6, 1, 7, 2, 3, 5,
       0, 4, 3, 3, 5, 0, 4, 3, 7, 6, 6, 1, 2, 5, 0, 4, 3, 5, 4, 3, 5, 0,
       4, 2, 2, 2, 1, 2, 6, 6, 2, 2, 7, 6, 1, 7, 2, 7, 4, 3, 5, 0, 4, 3,
       5, 0, 4, 3, 7, 1, 7, 7, 1, 1, 2, 6, 1, 2, 5, 0, 4, 3, 5, 7, 2, 6,
       6, 6, 4, 3, 5, 0, 4, 6, 7, 2, 2, 1, 5, 0, 4, 3, 5, 1, 1, 6, 1, 6,
       3, 5, 0, 4, 3, 6, 2, 7, 1, 1, 0, 4, 3, 5, 0, 7, 2, 7, 6, 1, 2, 2,
       2, 7, 1, 0, 4, 3, 5, 0, 0, 4, 3, 5, 0, 6, 6, 1, 1, 6, 6, 7, 1, 7,
       7, 3, 5, 0, 4, 3, 6, 1, 1, 6, 6, 7, 1, 2, 6, 1, 5, 0, 4, 3, 5, 3,
       5, 0, 4, 3, 6, 6, 2, 6, 6, 4, 3, 5, 0, 4, 2, 7, 6, 2, 7, 3, 5, 0,
       4, 3, 2, 1, 6, 1, 6, 0, 4, 3, 5, 0, 2, 7, 6, 1, 6, 7, 2, 1, 7, 6,
       0, 4, 3, 5, 0, 0, 4, 3, 5, 0, 1, 7, 1, 2, 2, 0, 4, 3, 5, 0, 5, 0,
       4, 3, 5, 2, 1, 7, 2, 2, 1, 7, 7, 2, 2, 4, 3, 5, 0, 4, 1, 6, 7, 6,
       1, 2, 6, 6, 1, 2, 4, 3, 5, 0, 4, 4, 3, 5, 0, 4, 0, 4, 3, 5, 0, 6,
       7, 2, 2, 7, 3, 5, 0, 4, 3, 1, 7, 7, 2, 6, 0, 4, 3, 5, 0, 4, 3, 5,
       0, 4, 2, 6, 1, 1, 2, 2, 6, 6, 7, 7, 5, 0, 4, 3, 5, 4, 3, 5, 0, 4,
       6, 2, 7, 6, 1, 3, 5, 0, 4, 3, 1, 7, 7, 6, 7, 0, 4, 3, 5, 0, 7, 2,
       2, 7, 6, 6, 7, 1, 2, 2, 0, 4, 3, 5, 0, 6, 2, 6, 2, 1, 4, 3, 5, 0,
       4, 6, 1, 1, 6, 1, 0, 4, 3, 5, 0, 6, 6, 7, 6, 1, 0, 4, 3, 5, 0, 1,
       6, 6, 7, 6, 0, 4, 3, 5, 0, 2, 6, 6, 7, 1, 5, 0, 4, 3, 5, 4, 3, 5,
       0, 4, 7, 6, 1, 6, 7, 2, 1, 6, 7, 7, 7, 1, 1, 2, 7, 4, 3, 5, 0, 4,
       6, 2, 2, 6, 6, 4, 3, 5, 0, 4, 6, 7, 7, 6, 6, 5, 0, 4, 3, 5, 7, 7,
       2, 2, 7, 1, 6, 6, 1, 2, 5, 0, 4, 3, 5, 5, 0, 4, 3, 5, 1, 1, 6, 6,
       6, 5, 0, 4, 3, 5, 3, 5, 0, 4, 3, 1, 2, 2, 1, 7, 1, 6, 1, 2, 2, 4,
       3, 5, 0, 4, 1, 2, 7, 6, 6, 4, 3, 5, 0, 4, 7, 1, 7, 2, 1, 4, 3, 5,
       0, 4, 3, 5, 0, 4, 3, 7, 1, 1, 2, 2, 1, 1, 2, 2, 2, 0, 4, 3, 5, 0,
       3, 5, 0, 4, 3, 7, 6, 1, 1, 6, 3, 5, 0, 4, 3, 2, 7, 1, 6, 7, 3, 5,
       0, 4, 3, 7, 7, 7, 1, 6, 1, 2, 6, 7, 7, 5, 0, 4, 3, 5, 7, 2, 7, 1,
       1, 5, 0, 4, 3, 5, 7, 1, 2, 7, 6, 5, 0, 4, 3, 5, 6, 2, 7, 7, 1, 5,
       0, 4, 3, 5, 6, 7, 6, 7, 1, 5, 0, 4, 3, 5, 2, 6, 2, 2, 2, 0, 4, 3,
       5, 0, 4, 3, 5, 0, 4, 1, 2, 1, 6, 6, 2, 7, 1, 6, 2, 0, 4, 3, 5, 0,
       2, 1, 6, 2, 2, 1, 7, 7, 2, 1, 3, 5, 0, 4, 3, 0, 4, 3, 5, 0, 6, 1,
       7, 6, 1, 3, 5, 0, 4, 3, 5, 0, 4, 3, 5, 6, 6, 2, 1, 2, 6, 1, 6, 2,
       1, 3, 5, 0, 4, 3, 7, 6, 7, 1, 2, 0, 4, 3, 5, 0, 2, 7, 6, 2, 7, 3,
       5, 0, 4, 3, 7, 2, 1, 1, 7, 5, 0, 4, 3, 5, 3, 5, 0, 4, 3, 1, 2, 1,
       7, 1, 1, 2, 7, 1, 7, 5, 0, 4, 3, 5, 1, 2, 2, 7, 7, 7, 1, 2, 7, 2,
       4, 3, 5, 0, 4, 5, 0, 4, 3, 5]
  mean_out_areas = []
  mean_in_areas = []
  for p in range(len(sym_list)):
    symbol = sym_list[p]
    row_vals = symbol_splitter(symbol)[1]
    clmn_vals = symbol_splitter(symbol)[2]
    out_row = row_vals[0] - 4, row_vals[1] + 4
    out_clmn = clmn_vals[0] - 4, clmn_vals[1] + 4
    in_row = row_vals[0] + 4, row_vals[1] - 4
    in_clmn = clmn_vals[0] + 4, clmn_vals[1] - 4
    #areas outside symbols:                                                                                             #slice around out_symbol to remove excess area with possible mark or symbol overlap
    out_symbol_1 = symbol[0:out_row[0], 0:(out_clmn[0])]
    out_symbol_1 = out_symbol_1[5:,5:]
    out_symbol_2 = symbol[0:out_row[0], out_clmn[0]:out_clmn[1]]                                                        #due to issues in later calculations change lower right edge grayscale values to white (255) see below
    out_symbol_2 = out_symbol_2[5:]
    out_symbol_3 = symbol[0:out_row[0], out_clmn[1]:-1]
    out_symbol_3 = out_symbol_3[5:,:-5]
    out_symbol_4 = symbol[out_row[0]:out_row[1], out_clmn[1]:-1]
    out_symbol_4 = out_symbol_4[:,:-5]
    out_symbol_5 = symbol[out_row[1]:-1, out_clmn[1]:-1]
    out_symbol_5 = out_symbol_5[:-5,:-5]
    out_symbol_6 = symbol[out_row[1]:-1, out_clmn[0]:out_clmn[1]]
    out_symbol_6 = out_symbol_6[:-5]
    out_symbol_7 = symbol[out_row[1]:-1, 0:out_clmn[0]]
    out_symbol_7 = out_symbol_7[:-5,5:]
    out_symbol_8 = symbol[out_row[0]:out_row[1], 0:out_clmn[0]]
    out_symbol_8 = out_symbol_8[:,5:]
    out_symbol = [out_symbol_1, out_symbol_2, out_symbol_3, out_symbol_4, out_symbol_5, out_symbol_6, out_symbol_7, out_symbol_8]
    #calculate means for each area and compare for lowest mean:
    mean_out_1 = np.mean(out_symbol_1)
    mean_out_2 = np.mean(out_symbol_2)
    mean_out_3 = np.mean(out_symbol_3)
    mean_out_4 = np.mean(out_symbol_4)
    mean_out_5 = np.mean(out_symbol_5)
    mean_out_6 = np.mean(out_symbol_6)
    mean_out_7 = np.mean(out_symbol_7)
    mean_out_8 = np.mean(out_symbol_8)
    mean_out = [mean_out_1, mean_out_2, mean_out_3, mean_out_4, mean_out_5, mean_out_6, mean_out_7, mean_out_8]
    if sym_var[p] == 2:                                                                                               # at index 2 is symbol variations 3
        del mean_out[1]  # delete and replace with new mean_out
        out_symbol_2[-5:, -5:] = 255  # change lower right edge of out_symbol to white for FP correction
        mean_out_2 = np.mean(out_symbol_2)
        mean_out.insert(1, mean_out_2)
        del mean_out[3]
        out_symbol_4[:3, :3] = 255
        mean_out_4 = np.mean(out_symbol_4)
        mean_out.insert(3, mean_out_4)
    elif sym_var[p] == 0:
        del mean_out[1]
        out_symbol_2[-3:, :3] = 255
        mean_out_2 = np.mean(out_symbol_2)
        mean_out.insert(1, mean_out_2)
        del mean_out[7]
        out_symbol_8[:3, -3:] = 255
        mean_out_8 = np.mean(out_symbol_8)
        mean_out.insert(7, mean_out_8)
    elif sym_var[p] == 4:
        del mean_out[5]
        out_symbol_6[:3, -3:] = 255
        mean_out_6 = np.mean(out_symbol_6)
        mean_out.insert(5, mean_out_6)
        del mean_out[3]
        out_symbol_4[-3:, :3] = 255
        mean_out_4 = np.mean(out_symbol_4)
        mean_out.insert(3, mean_out_4)
    elif sym_var[p] == 6:                                                                                               # at index 6 is symbol variations 7
        del mean_out[5]  # delete and replace with new mean_out
        out_symbol_6[:3, :3] = 255  # change upper right edge of out_symbol to white for FP correction
        mean_out_6 = np.mean(out_symbol_6)
        mean_out.insert(5, mean_out_6)
        del mean_out[7]
        out_symbol_8[-3:, -3:] = 255
        mean_out_8 = np.mean(out_symbol_8)
        mean_out.insert(7, mean_out_8)
    del mean_out[sym_var[p]]
    mean_out = np.floor(mean_out)
    mean_out_areas.append(mean_out)
    #areas inside symbol:
    in_symbol = symbol[in_row[0]:in_row[1], in_clmn[0]:in_clmn[1]]
    mean_in_sym = np.floor(np.mean(in_symbol))
    mean_in_areas.append(mean_in_sym)
  #check within symbol border for mark:
  mean_in_areas = np.array(mean_in_areas)
  mark_in = (mean_in_areas < thresh).astype(int)                                                                           #np.logical_and to combine two conditions without causing ValueError
  indx_mark_in = np.array(np.where(mark_in == 1)).flatten()
  #use areas outside the symbol to check for symbols without a definite mark:
  mean_out_areas = np.array(mean_out_areas)
  indx_mark_out = np.array(np.where(mark_in == 0)).flatten()                                                            #index for symbols that weren't recognized as marked
  mark_out = np.zeros(1000, int)
  for k in indx_mark_out:
    if np.count_nonzero(mean_out_areas[k] < thresh) > 1:                                                                   #whenever two or more areas outside the symbol show a mark it is considered marked
      mark_out[k] = 1
    else:
      mark_out[k] = 0
  #join mark arrays:
  mark = np.add(mark_out, mark_in)
  mark[0:25] = 0
  indx_mark_sym = np.array(np.where(mark == 1)).flatten()
  if corr == True:
      mark_sym = np.array(mean_sym_list)[indx_mark_sym]
      mean_mark_sym = np.mean(mark_sym)  # uncomment for mean/std correction control
      std_mark_sym = np.std(mark_sym)
      indx_outl = np.where(mark_sym < (mean_mark_sym - 3 * std_mark_sym))
      indx_corr = indx_mark_sym[indx_outl]
      mark[indx_corr] = 0
      mark = mark.reshape(40, 25)
  else:
      mark = mark.reshape(40, 25)
  return mark, mean_in_areas

def model_in_AND_out(sym_list, mean_sym_list, thresh = 250, corr = False):
  #symbol variations generated from empty test sheet:
  sym_var = [4, 3, 5, 0, 4, 7, 6, 1, 1, 7, 3, 5, 0, 4, 3, 7, 1, 7, 1, 2, 4, 3,
             5, 0, 4, 1, 7, 7, 1, 1, 4, 3, 5, 0, 4, 5, 0, 4, 3, 5, 6, 2, 6, 6,
             7, 1, 2, 6, 6, 1, 3, 5, 0, 4, 3, 1, 7, 6, 2, 1, 7, 1, 1, 7, 6, 0,
             4, 3, 5, 0, 3, 5, 0, 4, 3, 7, 7, 2, 7, 7, 4, 3, 5, 0, 4, 1, 6, 2,
             1, 6, 3, 5, 0, 4, 3, 7, 6, 2, 7, 7, 0, 4, 3, 5, 0, 2, 2, 6, 7, 2,
             0, 4, 3, 5, 0, 7, 6, 6, 7, 6, 0, 4, 3, 5, 0, 4, 3, 5, 0, 4, 1, 1,
             2, 7, 1, 5, 0, 4, 3, 5, 2, 7, 2, 6, 7, 4, 3, 5, 0, 4, 2, 7, 2, 6,
             2, 3, 5, 0, 4, 3, 6, 7, 1, 6, 6, 4, 3, 5, 0, 4, 2, 6, 2, 2, 2, 5,
             0, 4, 3, 5, 6, 7, 2, 1, 6, 2, 1, 7, 2, 7, 5, 0, 4, 3, 5, 5, 0, 4,
             3, 5, 2, 2, 7, 7, 6, 0, 4, 3, 5, 0, 0, 4, 3, 5, 0, 2, 1, 1, 6, 2,
             2, 6, 1, 1, 1, 3, 5, 0, 4, 3, 2, 1, 1, 7, 7, 6, 6, 1, 7, 2, 3, 5,
             0, 4, 3, 3, 5, 0, 4, 3, 7, 6, 6, 1, 2, 5, 0, 4, 3, 5, 4, 3, 5, 0,
             4, 2, 2, 2, 1, 2, 6, 6, 2, 2, 7, 6, 1, 7, 2, 7, 4, 3, 5, 0, 4, 3,
             5, 0, 4, 3, 7, 1, 7, 7, 1, 1, 2, 6, 1, 2, 5, 0, 4, 3, 5, 7, 2, 6,
             6, 6, 4, 3, 5, 0, 4, 6, 7, 2, 2, 1, 5, 0, 4, 3, 5, 1, 1, 6, 1, 6,
             3, 5, 0, 4, 3, 6, 2, 7, 1, 1, 0, 4, 3, 5, 0, 7, 2, 7, 6, 1, 2, 2,
             2, 7, 1, 0, 4, 3, 5, 0, 0, 4, 3, 5, 0, 6, 6, 1, 1, 6, 6, 7, 1, 7,
             7, 3, 5, 0, 4, 3, 6, 1, 1, 6, 6, 7, 1, 2, 6, 1, 5, 0, 4, 3, 5, 3,
             5, 0, 4, 3, 6, 6, 2, 6, 6, 4, 3, 5, 0, 4, 2, 7, 6, 2, 7, 3, 5, 0,
             4, 3, 2, 1, 6, 1, 6, 0, 4, 3, 5, 0, 2, 7, 6, 1, 6, 7, 2, 1, 7, 6,
             0, 4, 3, 5, 0, 0, 4, 3, 5, 0, 1, 7, 1, 2, 2, 0, 4, 3, 5, 0, 5, 0,
             4, 3, 5, 2, 1, 7, 2, 2, 1, 7, 7, 2, 2, 4, 3, 5, 0, 4, 1, 6, 7, 6,
             1, 2, 6, 6, 1, 2, 4, 3, 5, 0, 4, 4, 3, 5, 0, 4, 0, 4, 3, 5, 0, 6,
             7, 2, 2, 7, 3, 5, 0, 4, 3, 1, 7, 7, 2, 6, 0, 4, 3, 5, 0, 4, 3, 5,
             0, 4, 2, 6, 1, 1, 2, 2, 6, 6, 7, 7, 5, 0, 4, 3, 5, 4, 3, 5, 0, 4,
             6, 2, 7, 6, 1, 3, 5, 0, 4, 3, 1, 7, 7, 6, 7, 0, 4, 3, 5, 0, 7, 2,
             2, 7, 6, 6, 7, 1, 2, 2, 0, 4, 3, 5, 0, 6, 2, 6, 2, 1, 4, 3, 5, 0,
             4, 6, 1, 1, 6, 1, 0, 4, 3, 5, 0, 6, 6, 7, 6, 1, 0, 4, 3, 5, 0, 1,
             6, 6, 7, 6, 0, 4, 3, 5, 0, 2, 6, 6, 7, 1, 5, 0, 4, 3, 5, 4, 3, 5,
             0, 4, 7, 6, 1, 6, 7, 2, 1, 6, 7, 7, 7, 1, 1, 2, 7, 4, 3, 5, 0, 4,
             6, 2, 2, 6, 6, 4, 3, 5, 0, 4, 6, 7, 7, 6, 6, 5, 0, 4, 3, 5, 7, 7,
             2, 2, 7, 1, 6, 6, 1, 2, 5, 0, 4, 3, 5, 5, 0, 4, 3, 5, 1, 1, 6, 6,
             6, 5, 0, 4, 3, 5, 3, 5, 0, 4, 3, 1, 2, 2, 1, 7, 1, 6, 1, 2, 2, 4,
             3, 5, 0, 4, 1, 2, 7, 6, 6, 4, 3, 5, 0, 4, 7, 1, 7, 2, 1, 4, 3, 5,
             0, 4, 3, 5, 0, 4, 3, 7, 1, 1, 2, 2, 1, 1, 2, 2, 2, 0, 4, 3, 5, 0,
             3, 5, 0, 4, 3, 7, 6, 1, 1, 6, 3, 5, 0, 4, 3, 2, 7, 1, 6, 7, 3, 5,
             0, 4, 3, 7, 7, 7, 1, 6, 1, 2, 6, 7, 7, 5, 0, 4, 3, 5, 7, 2, 7, 1,
             1, 5, 0, 4, 3, 5, 7, 1, 2, 7, 6, 5, 0, 4, 3, 5, 6, 2, 7, 7, 1, 5,
             0, 4, 3, 5, 6, 7, 6, 7, 1, 5, 0, 4, 3, 5, 2, 6, 2, 2, 2, 0, 4, 3,
             5, 0, 4, 3, 5, 0, 4, 1, 2, 1, 6, 6, 2, 7, 1, 6, 2, 0, 4, 3, 5, 0,
             2, 1, 6, 2, 2, 1, 7, 7, 2, 1, 3, 5, 0, 4, 3, 0, 4, 3, 5, 0, 6, 1,
             7, 6, 1, 3, 5, 0, 4, 3, 5, 0, 4, 3, 5, 6, 6, 2, 1, 2, 6, 1, 6, 2,
             1, 3, 5, 0, 4, 3, 7, 6, 7, 1, 2, 0, 4, 3, 5, 0, 2, 7, 6, 2, 7, 3,
             5, 0, 4, 3, 7, 2, 1, 1, 7, 5, 0, 4, 3, 5, 3, 5, 0, 4, 3, 1, 2, 1,
             7, 1, 1, 2, 7, 1, 7, 5, 0, 4, 3, 5, 1, 2, 2, 7, 7, 7, 1, 2, 7, 2,
             4, 3, 5, 0, 4, 5, 0, 4, 3, 5]
  #splitting the symbols:
  mean_out_areas = []
  mean_in_areas = []
  for p in range(len(sym_list)):
    symbol = sym_list[p]
    row_vals = symbol_splitter(symbol)[1]
    clmn_vals = symbol_splitter(symbol)[2]
    out_row = row_vals[0] - 4, row_vals[1] + 4
    out_clmn = clmn_vals[0] - 4, clmn_vals[1] + 4
    in_row = row_vals[0] + 4, row_vals[1] - 4
    in_clmn = clmn_vals[0] + 4, clmn_vals[1] - 4
    #areas outside symbols:                                                                                             #slice around out_symbol to remove excess area with possible mark or symbol overlap
    out_symbol_1 = symbol[0:out_row[0], 0:(out_clmn[0])]
    out_symbol_1 = out_symbol_1[5:,5:]
    out_symbol_2 = symbol[0:out_row[0], out_clmn[0]:out_clmn[1]]                                                        #due to issues in later calculations change lower right edge grayscale values to white (255) see below
    out_symbol_2 = out_symbol_2[5:]
    out_symbol_3 = symbol[0:out_row[0], out_clmn[1]:-1]
    out_symbol_3 = out_symbol_3[5:,:-5]
    out_symbol_4 = symbol[out_row[0]:out_row[1], out_clmn[1]:-1]
    out_symbol_4 = out_symbol_4[:,:-5]
    out_symbol_5 = symbol[out_row[1]:-1, out_clmn[1]:-1]
    out_symbol_5 = out_symbol_5[:-5,:-5]
    out_symbol_6 = symbol[out_row[1]:-1, out_clmn[0]:out_clmn[1]]
    out_symbol_6 = out_symbol_6[:-5]
    out_symbol_7 = symbol[out_row[1]:-1, 0:out_clmn[0]]
    out_symbol_7 = out_symbol_7[:-5,5:]
    out_symbol_8 = symbol[out_row[0]:out_row[1], 0:out_clmn[0]]
    out_symbol_8 = out_symbol_8[:,5:]
    out_symbol = [out_symbol_1, out_symbol_2, out_symbol_3, out_symbol_4, out_symbol_5, out_symbol_6, out_symbol_7, out_symbol_8]
    #calculate means for each area and compare for lowest mean:
    mean_out_1 = np.mean(out_symbol_1)
    mean_out_2 = np.mean(out_symbol_2)
    mean_out_3 = np.mean(out_symbol_3)
    mean_out_4 = np.mean(out_symbol_4)
    mean_out_5 = np.mean(out_symbol_5)
    mean_out_6 = np.mean(out_symbol_6)
    mean_out_7 = np.mean(out_symbol_7)
    mean_out_8 = np.mean(out_symbol_8)
    mean_out = [mean_out_1, mean_out_2, mean_out_3, mean_out_4, mean_out_5, mean_out_6, mean_out_7, mean_out_8]
    if sym_var[p] == 2:  # at index 2 is symbol variations 3
        del mean_out[1]  # delete and replace with new mean_out
        out_symbol_2[-5:, -5:] = 255  # change lower right edge of out_symbol to white for FP correction
        mean_out_2 = np.mean(out_symbol_2)
        mean_out.insert(1, mean_out_2)
        del mean_out[3]
        out_symbol_4[:3, :3] = 255
        mean_out_4 = np.mean(out_symbol_4)
        mean_out.insert(3, mean_out_4)
    elif sym_var[p] == 0:
        del mean_out[1]
        out_symbol_2[-3:, :3] = 255
        mean_out_2 = np.mean(out_symbol_2)
        mean_out.insert(1, mean_out_2)
        del mean_out[7]
        out_symbol_8[:3, -3:] = 255
        mean_out_8 = np.mean(out_symbol_8)
        mean_out.insert(7, mean_out_8)
    elif sym_var[p] == 4:
        del mean_out[5]
        out_symbol_6[:3, -3:] = 255
        mean_out_6 = np.mean(out_symbol_6)
        mean_out.insert(5, mean_out_6)
        del mean_out[3]
        out_symbol_4[-3:, :3] = 255
        mean_out_4 = np.mean(out_symbol_4)
        mean_out.insert(3, mean_out_4)
    elif sym_var[p] == 6:  # at index 6 is symbol variations 7
        del mean_out[5]  # delete and replace with new mean_out
        out_symbol_6[:3, :3] = 255  # change upper right edge of out_symbol to white for FP correction
        mean_out_6 = np.mean(out_symbol_6)
        mean_out.insert(5, mean_out_6)
        del mean_out[7]
        out_symbol_8[-3:, -3:] = 255
        mean_out_8 = np.mean(out_symbol_8)
        mean_out.insert(7, mean_out_8)
    del mean_out[sym_var[p]]
    mean_out = np.floor(mean_out)
    mean_out_areas.append(mean_out)
    #areas inside symbol:
    in_symbol = symbol[in_row[0]:in_row[1], in_clmn[0]:in_clmn[1]]
    mean_in_sym = np.floor(np.mean(in_symbol))
    mean_in_areas.append(mean_in_sym)
  #check within symbol border for mark:
  mean_in_areas = np.array(mean_in_areas)
  mark_in = (mean_in_areas <= thresh).astype(int)                                                                               #np.logical_and to combine two conditions without causing ValueError
  indx_mark_in = np.array(np.where(mark_in == 1)).flatten()
  #use areas outside the symbol to check for symbols without a definite mark:
  mean_out_areas = np.array(mean_out_areas)
  mark_out = np.zeros(1000, int)
  for k in range(len(mean_in_areas)):
    if np.count_nonzero(mean_out_areas[k] <= thresh) > 0:
        mark_out[k] = 1
    else:
        mark_out[k] = 0
  indx_mark_out = np.array(np.where(mark_out == 1)).flatten()                                                           #index for symbols that weren't recognized as marked inside the symbol
  #join mark arrays:
  mark = np.add(mark_out, mark_in)
  mark[mark == 1] = 0
  mark[mark == 2] = 1
  mark[0:25] = 0
  indx_mark_sym = np.array(np.where(mark == 1)).flatten()
  if corr == True:
    mark_sym = np.array(mean_sym_list)[indx_mark_sym]
    mean_mark_sym = np.mean(mark_sym)                                                                                   #uncomment for mean/std correction control
    std_mark_sym = np.std(mark_sym)
    indx_outl = np.where(mark_sym < (mean_mark_sym - 3*std_mark_sym))
    indx_corr = indx_mark_sym[indx_outl]
    mark[indx_corr] = 0
    mark = mark.reshape(40, 25)
  else:
    mark = mark.reshape(40, 25)
  return mark, mean_in_areas

def model_strict_thresh(sym_list, mean_sym_list, thresh = 250, strict_thresh = 230, corr = False):
  #symbol variations generated from empty test sheet:
  sym_var = [4, 3, 5, 0, 4, 7, 6, 1, 1, 7, 3, 5, 0, 4, 3, 7, 1, 7, 1, 2, 4, 3,
             5, 0, 4, 1, 7, 7, 1, 1, 4, 3, 5, 0, 4, 5, 0, 4, 3, 5, 6, 2, 6, 6,
             7, 1, 2, 6, 6, 1, 3, 5, 0, 4, 3, 1, 7, 6, 2, 1, 7, 1, 1, 7, 6, 0,
             4, 3, 5, 0, 3, 5, 0, 4, 3, 7, 7, 2, 7, 7, 4, 3, 5, 0, 4, 1, 6, 2,
             1, 6, 3, 5, 0, 4, 3, 7, 6, 2, 7, 7, 0, 4, 3, 5, 0, 2, 2, 6, 7, 2,
             0, 4, 3, 5, 0, 7, 6, 6, 7, 6, 0, 4, 3, 5, 0, 4, 3, 5, 0, 4, 1, 1,
             2, 7, 1, 5, 0, 4, 3, 5, 2, 7, 2, 6, 7, 4, 3, 5, 0, 4, 2, 7, 2, 6,
             2, 3, 5, 0, 4, 3, 6, 7, 1, 6, 6, 4, 3, 5, 0, 4, 2, 6, 2, 2, 2, 5,
             0, 4, 3, 5, 6, 7, 2, 1, 6, 2, 1, 7, 2, 7, 5, 0, 4, 3, 5, 5, 0, 4,
             3, 5, 2, 2, 7, 7, 6, 0, 4, 3, 5, 0, 0, 4, 3, 5, 0, 2, 1, 1, 6, 2,
             2, 6, 1, 1, 1, 3, 5, 0, 4, 3, 2, 1, 1, 7, 7, 6, 6, 1, 7, 2, 3, 5,
             0, 4, 3, 3, 5, 0, 4, 3, 7, 6, 6, 1, 2, 5, 0, 4, 3, 5, 4, 3, 5, 0,
             4, 2, 2, 2, 1, 2, 6, 6, 2, 2, 7, 6, 1, 7, 2, 7, 4, 3, 5, 0, 4, 3,
             5, 0, 4, 3, 7, 1, 7, 7, 1, 1, 2, 6, 1, 2, 5, 0, 4, 3, 5, 7, 2, 6,
             6, 6, 4, 3, 5, 0, 4, 6, 7, 2, 2, 1, 5, 0, 4, 3, 5, 1, 1, 6, 1, 6,
             3, 5, 0, 4, 3, 6, 2, 7, 1, 1, 0, 4, 3, 5, 0, 7, 2, 7, 6, 1, 2, 2,
             2, 7, 1, 0, 4, 3, 5, 0, 0, 4, 3, 5, 0, 6, 6, 1, 1, 6, 6, 7, 1, 7,
             7, 3, 5, 0, 4, 3, 6, 1, 1, 6, 6, 7, 1, 2, 6, 1, 5, 0, 4, 3, 5, 3,
             5, 0, 4, 3, 6, 6, 2, 6, 6, 4, 3, 5, 0, 4, 2, 7, 6, 2, 7, 3, 5, 0,
             4, 3, 2, 1, 6, 1, 6, 0, 4, 3, 5, 0, 2, 7, 6, 1, 6, 7, 2, 1, 7, 6,
             0, 4, 3, 5, 0, 0, 4, 3, 5, 0, 1, 7, 1, 2, 2, 0, 4, 3, 5, 0, 5, 0,
             4, 3, 5, 2, 1, 7, 2, 2, 1, 7, 7, 2, 2, 4, 3, 5, 0, 4, 1, 6, 7, 6,
             1, 2, 6, 6, 1, 2, 4, 3, 5, 0, 4, 4, 3, 5, 0, 4, 0, 4, 3, 5, 0, 6,
             7, 2, 2, 7, 3, 5, 0, 4, 3, 1, 7, 7, 2, 6, 0, 4, 3, 5, 0, 4, 3, 5,
             0, 4, 2, 6, 1, 1, 2, 2, 6, 6, 7, 7, 5, 0, 4, 3, 5, 4, 3, 5, 0, 4,
             6, 2, 7, 6, 1, 3, 5, 0, 4, 3, 1, 7, 7, 6, 7, 0, 4, 3, 5, 0, 7, 2,
             2, 7, 6, 6, 7, 1, 2, 2, 0, 4, 3, 5, 0, 6, 2, 6, 2, 1, 4, 3, 5, 0,
             4, 6, 1, 1, 6, 1, 0, 4, 3, 5, 0, 6, 6, 7, 6, 1, 0, 4, 3, 5, 0, 1,
             6, 6, 7, 6, 0, 4, 3, 5, 0, 2, 6, 6, 7, 1, 5, 0, 4, 3, 5, 4, 3, 5,
             0, 4, 7, 6, 1, 6, 7, 2, 1, 6, 7, 7, 7, 1, 1, 2, 7, 4, 3, 5, 0, 4,
             6, 2, 2, 6, 6, 4, 3, 5, 0, 4, 6, 7, 7, 6, 6, 5, 0, 4, 3, 5, 7, 7,
             2, 2, 7, 1, 6, 6, 1, 2, 5, 0, 4, 3, 5, 5, 0, 4, 3, 5, 1, 1, 6, 6,
             6, 5, 0, 4, 3, 5, 3, 5, 0, 4, 3, 1, 2, 2, 1, 7, 1, 6, 1, 2, 2, 4,
             3, 5, 0, 4, 1, 2, 7, 6, 6, 4, 3, 5, 0, 4, 7, 1, 7, 2, 1, 4, 3, 5,
             0, 4, 3, 5, 0, 4, 3, 7, 1, 1, 2, 2, 1, 1, 2, 2, 2, 0, 4, 3, 5, 0,
             3, 5, 0, 4, 3, 7, 6, 1, 1, 6, 3, 5, 0, 4, 3, 2, 7, 1, 6, 7, 3, 5,
             0, 4, 3, 7, 7, 7, 1, 6, 1, 2, 6, 7, 7, 5, 0, 4, 3, 5, 7, 2, 7, 1,
             1, 5, 0, 4, 3, 5, 7, 1, 2, 7, 6, 5, 0, 4, 3, 5, 6, 2, 7, 7, 1, 5,
             0, 4, 3, 5, 6, 7, 6, 7, 1, 5, 0, 4, 3, 5, 2, 6, 2, 2, 2, 0, 4, 3,
             5, 0, 4, 3, 5, 0, 4, 1, 2, 1, 6, 6, 2, 7, 1, 6, 2, 0, 4, 3, 5, 0,
             2, 1, 6, 2, 2, 1, 7, 7, 2, 1, 3, 5, 0, 4, 3, 0, 4, 3, 5, 0, 6, 1,
             7, 6, 1, 3, 5, 0, 4, 3, 5, 0, 4, 3, 5, 6, 6, 2, 1, 2, 6, 1, 6, 2,
             1, 3, 5, 0, 4, 3, 7, 6, 7, 1, 2, 0, 4, 3, 5, 0, 2, 7, 6, 2, 7, 3,
             5, 0, 4, 3, 7, 2, 1, 1, 7, 5, 0, 4, 3, 5, 3, 5, 0, 4, 3, 1, 2, 1,
             7, 1, 1, 2, 7, 1, 7, 5, 0, 4, 3, 5, 1, 2, 2, 7, 7, 7, 1, 2, 7, 2,
             4, 3, 5, 0, 4, 5, 0, 4, 3, 5]
  #using template to generate target and non-target indices:
  x = [1,1,1,1,1,1,
      2,2,2,2,2,2,
      3,3,3,3,3,3,3,3,3,3,
      4,4,4,4,4,4,
      5,5,5,5,5,5,5,5,
      6,6,6,6,6,
      7,7,7,7,7,7,
      8,8,8,8,
      9,9,9,9,9,9,
      10,10,10,10,10,
      11,11,11,11,11,11,11,11,
      12,12,12,12,12,12,
      13,13,13,13,13,
      14,14,14,14,14,14,
      15,15,15,15,
      16,16,16,16,16,
      17,17,17,17,17,17,
      18,18,18,18,18,18,
      19,19,19,19,19,19,19,
      20,20,20,20,20,20,20,
      21,21,21,21,21,21,21,
      22,22,22,22,22,22,22,22,
      23,23,23,23,
      24,24,24,24,24,
      25,25,25,25,25,25,25,25,
      26,26,26,26,26,26,26,26,
      27,27,27,27,27,27,
      28,28,28,
      29,29,29,29,29,29,29,29,29,
      30,30,30,30,
      31,31,31,31,31,31,31,31,
      32,32,32,32,32,32,32,32,
      33,33,33,33,33,33,33,
      34,34,34,34,
      35,35,35,35,35,
      36,36,36,
      37,37,37,37,37,37,37,
      38,38,38,38,38,38,38,
      39,39,39,39,39,39,39,39]      #rows
  y = [1,2,5,9,12,19,
      3,6,10,13,16,23,
      0,1,3,4,5,9,18,20,23,24,
      1,8,11,15,18,21,
      0,4,8,12,16,19,20,24,
      1,8,11,15,19,
      2,6,12,14,17,22,
      2,3,6,11,
      3,8,9,13,18,23,
      0,7,10,14,24,
      2,4,5,9,13,15,17,18,
      2,5,10,14,16,22,
      8,12,16,20,22,
      3,6,11,21,23,24,
      3,10,17,23,
      5,9,11,14,18,
      1,6,10,13,16,21,
      1,6,12,17,21,22,
      0,4,7,15,19,20,24,
      1,6,9,13,16,17,21,
      0,4,13,14,17,20,24,
      2,8,11,12,14,16,20,23,
      1,6,15,19,
      1,7,11,18,21,
      3,7,10,14,15,19,23,24,
      0,4,5,9,15,19,21,22,
      2,5,6,9,17,22,
      7,13,19,
      0,4,7,10,14,15,17,20,24,
      3,5,16,23,
      0,8,11,14,18,20,21,22,
      3,4,7,10,12,17,20,23,
      2,7,8,12,16,18,22,
      6,10,14,21,
      1,11,12,18,21,
      2,8,12,
      3,5,7,11,16,19,23,
      0,4,7,13,18,22,24,
      2,8,9,10,13,15,19,22]      #columns
  def listmaker(n):
      listofones = [1] * n
      return listofones
  value = listmaker(241)        #list of ones for each true positive position
  row = np.array(x)
  col = np.array(y)
  data = np.array(value)
  template = csr_matrix((data, (row, col)), shape=(40, 25)).toarray()
  template = template.flatten()
  indx_target = np.array(np.where(template == 1)).flatten()
  indx_non_target = np.array(np.where(template == 0)).flatten()
  #splitting the symbols:
  mean_out_areas = []
  mean_in_areas = []
  for p in range(len(sym_list)):
    symbol = sym_list[p]
    row_vals = symbol_splitter(symbol)[1]
    clmn_vals = symbol_splitter(symbol)[2]
    out_row = row_vals[0] - 4, row_vals[1] + 4
    out_clmn = clmn_vals[0] - 4, clmn_vals[1] + 4
    in_row = row_vals[0] + 4, row_vals[1] - 4
    in_clmn = clmn_vals[0] + 4, clmn_vals[1] - 4
    #areas outside symbols:                                                                                             #slice around out_symbol to remove excess area with possible mark or symbol overlap
    out_symbol_1 = symbol[0:out_row[0], 0:(out_clmn[0])]
    out_symbol_1 = out_symbol_1[5:,5:]
    out_symbol_2 = symbol[0:out_row[0], out_clmn[0]:out_clmn[1]]                                                        #due to issues in later calculations change lower right edge grayscale values to white (255) see below
    out_symbol_2 = out_symbol_2[5:]
    out_symbol_3 = symbol[0:out_row[0], out_clmn[1]:-1]
    out_symbol_3 = out_symbol_3[5:,:-5]
    out_symbol_4 = symbol[out_row[0]:out_row[1], out_clmn[1]:-1]
    out_symbol_4 = out_symbol_4[:,:-5]
    out_symbol_5 = symbol[out_row[1]:-1, out_clmn[1]:-1]
    out_symbol_5 = out_symbol_5[:-5,:-5]
    out_symbol_6 = symbol[out_row[1]:-1, out_clmn[0]:out_clmn[1]]
    out_symbol_6 = out_symbol_6[:-5]
    out_symbol_7 = symbol[out_row[1]:-1, 0:out_clmn[0]]
    out_symbol_7 = out_symbol_7[:-5,5:]
    out_symbol_8 = symbol[out_row[0]:out_row[1], 0:out_clmn[0]]
    out_symbol_8 = out_symbol_8[:,5:]
    out_symbol = [out_symbol_1, out_symbol_2, out_symbol_3, out_symbol_4, out_symbol_5, out_symbol_6, out_symbol_7, out_symbol_8]
    #calculate means for each area and compare for lowest mean:
    mean_out_1 = np.mean(out_symbol_1)
    mean_out_2 = np.mean(out_symbol_2)
    mean_out_3 = np.mean(out_symbol_3)
    mean_out_4 = np.mean(out_symbol_4)
    mean_out_5 = np.mean(out_symbol_5)
    mean_out_6 = np.mean(out_symbol_6)
    mean_out_7 = np.mean(out_symbol_7)
    mean_out_8 = np.mean(out_symbol_8)
    mean_out = [mean_out_1, mean_out_2, mean_out_3, mean_out_4, mean_out_5, mean_out_6, mean_out_7, mean_out_8]
    if sym_var[p] == 2:                                                                                                 # at index 2 is symbol variations 3
        del mean_out[1]                                                                                                 # delete and replace with new mean_out
        out_symbol_2[-5:, -5:] = 255                                                                                    # change lower right edge of out_symbol to white for FP correction
        mean_out_2 = np.mean(out_symbol_2)
        mean_out.insert(1, mean_out_2)
        del mean_out[3]
        out_symbol_4[:3, :3] = 255
        mean_out_4 = np.mean(out_symbol_4)
        mean_out.insert(3, mean_out_4)
    elif sym_var[p] == 0:
        del mean_out[1]
        out_symbol_2[-3:, :3] = 255
        mean_out_2 = np.mean(out_symbol_2)
        mean_out.insert(1, mean_out_2)
        del mean_out[7]
        out_symbol_8[:3, -3:] = 255
        mean_out_8 = np.mean(out_symbol_8)
        mean_out.insert(7, mean_out_8)
    elif sym_var[p] == 4:
        del mean_out[5]
        out_symbol_6[:3, -3:] = 255
        mean_out_6 = np.mean(out_symbol_6)
        mean_out.insert(5, mean_out_6)
        del mean_out[3]
        out_symbol_4[-3:, :3] = 255
        mean_out_4 = np.mean(out_symbol_4)
        mean_out.insert(3, mean_out_4)
    elif sym_var[p] == 6:                                                                                               # at index 6 is symbol variations 7
        del mean_out[5]                                                                                                 # delete and replace with new mean_out
        out_symbol_6[:3, :3] = 255                                                                                      # change upper right edge of out_symbol to white for FP correction
        mean_out_6 = np.mean(out_symbol_6)
        mean_out.insert(5, mean_out_6)
        del mean_out[7]
        out_symbol_8[-3:, -3:] = 255
        mean_out_8 = np.mean(out_symbol_8)
        mean_out.insert(7, mean_out_8)
    del mean_out[sym_var[p]]
    mean_out = np.floor(mean_out)
    mean_out_areas.append(mean_out)
    #areas inside symbol:
    in_symbol = symbol[in_row[0]:in_row[1], in_clmn[0]:in_clmn[1]]
    mean_in_sym = np.floor(np.mean(in_symbol))
    mean_in_areas.append(mean_in_sym)
  #check within symbol border for target mark with lenient threshold:
  mean_in_areas = np.array(mean_in_areas)
  mark_in_target = np.zeros(1000, int)
  for k in indx_target:
    if mean_in_areas[k] <= thresh:
        mark_in_target[k] = 1
    else:
        mark_in_target[k] = 0
  #indx_mark_in_target = np.array(np.where(mark_in_target == 1)).flatten()
  #use areas outside the symbol to check for target symbols without a definite mark:
  mean_out_areas = np.array(mean_out_areas)
  indx_mark_out_target = np.intersect1d(indx_target, np.array(np.where(mark_in_target == 0)).flatten())        #index for symbols that weren't recognized as marked among target symbols
  mark_out_target = np.zeros(1000, int)
  for k in indx_mark_out_target:
    if np.count_nonzero(mean_out_areas[k] <= thresh) > 0:
        mark_out_target[k] = 1
    else:
        mark_out_target[k] = 0
  #indx_mark_out_target = np.array(np.where(mark_out_target == 1)).flatten()      #index for symbols that weren't recognized as marked inside the symbol
  #check within symbol border for non-target mark with strict threshold:
  mark_in_non_target = np.zeros(1000, int)
  for k in indx_non_target:
    if mean_in_areas[k] < strict_thresh:
        mark_in_non_target[k] = 1
    else:
        mark_in_non_target[k] = 0
  #use areas outside the symbol to check for non-target symbols without a definite mark:
  indx_mark_out_non_target = np.intersect1d(indx_non_target, np.array(np.where(mark_in_non_target == 0)).flatten())    #index for symbols that weren't recognized as marked
  mark_out_non_target = np.zeros(1000, int)
  for k in indx_mark_out_non_target:
    if np.count_nonzero(mean_out_areas[k] < strict_thresh) > 1:
        mark_out_non_target[k] = 1
    else:
        mark_out_non_target[k] = 0
  #join mark arrays:
  mark_target = np.add(mark_out_target, mark_in_target)
  mark_non_target = np.add(mark_out_non_target, mark_in_non_target)
  mark = np.add(mark_non_target, mark_target)
  mark[0:25] = 0
  indx_mark_sym = np.array(np.where(mark == 1)).flatten()
  if corr == True:
    mark_sym = np.array(mean_sym_list)[indx_mark_sym]
    mean_mark_sym = np.mean(mark_sym)                                                                                   #uncomment for mean/std correction control
    std_mark_sym = np.std(mark_sym)
    indx_outl = np.where(mark_sym < (mean_mark_sym - 3*std_mark_sym))
    indx_corr = indx_mark_sym[indx_outl]
    mark[indx_corr] = 0
    mark = mark.reshape(40, 25)
  else:
    mark = mark.reshape(40, 25)
  return mark, mean_in_areas

def adapt_template(stop):
  #template:
  x = [1,1,1,1,1,1,
      2,2,2,2,2,2,
      3,3,3,3,3,3,3,3,3,3,
      4,4,4,4,4,4,
      5,5,5,5,5,5,5,5,
      6,6,6,6,6,
      7,7,7,7,7,7,
      8,8,8,8,
      9,9,9,9,9,9,
      10,10,10,10,10,
      11,11,11,11,11,11,11,11,
      12,12,12,12,12,12,
      13,13,13,13,13,
      14,14,14,14,14,14,
      15,15,15,15,
      16,16,16,16,16,
      17,17,17,17,17,17,
      18,18,18,18,18,18,
      19,19,19,19,19,19,19,
      20,20,20,20,20,20,20,
      21,21,21,21,21,21,21,
      22,22,22,22,22,22,22,22,
      23,23,23,23,
      24,24,24,24,24,
      25,25,25,25,25,25,25,25,
      26,26,26,26,26,26,26,26,
      27,27,27,27,27,27,
      28,28,28,
      29,29,29,29,29,29,29,29,29,
      30,30,30,30,
      31,31,31,31,31,31,31,31,
      32,32,32,32,32,32,32,32,
      33,33,33,33,33,33,33,
      34,34,34,34,
      35,35,35,35,35,
      36,36,36,
      37,37,37,37,37,37,37,
      38,38,38,38,38,38,38,
      39,39,39,39,39,39,39,39]                                                                                          #rows
  y = [1,2,5,9,12,19,
      3,6,10,13,16,23,
      0,1,3,4,5,9,18,20,23,24,
      1,8,11,15,18,21,
      0,4,8,12,16,19,20,24,
      1,8,11,15,19,
      2,6,12,14,17,22,
      2,3,6,11,
      3,8,9,13,18,23,
      0,7,10,14,24,
      2,4,5,9,13,15,17,18,
      2,5,10,14,16,22,
      8,12,16,20,22,
      3,6,11,21,23,24,
      3,10,17,23,
      5,9,11,14,18,
      1,6,10,13,16,21,
      1,6,12,17,21,22,
      0,4,7,15,19,20,24,
      1,6,9,13,16,17,21,
      0,4,13,14,17,20,24,
      2,8,11,12,14,16,20,23,
      1,6,15,19,
      1,7,11,18,21,
      3,7,10,14,15,19,23,24,
      0,4,5,9,15,19,21,22,
      2,5,6,9,17,22,
      7,13,19,
      0,4,7,10,14,15,17,20,24,
      3,5,16,23,
      0,8,11,14,18,20,21,22,
      3,4,7,10,12,17,20,23,
      2,7,8,12,16,18,22,
      6,10,14,21,
      1,11,12,18,21,
      2,8,12,
      3,5,7,11,16,19,23,
      0,4,7,13,18,22,24,
      2,8,9,10,13,15,19,22]                                                                                             #columns
  def listmaker(n):
      listofones = [1] * n
      return listofones
  value = listmaker(241)                                                                                                #list of ones for each true positive position
  row = np.array(x)
  col = np.array(y)
  data = np.array(value)
  template = csr_matrix((data, (row, col)), shape=(40, 25)).toarray()
  stop_indx = np.array(((stop[0] - 1) * 25) + (stop[1] - 1))
  template = template.flatten()
  template[(stop_indx + 1):] = 0                                                                                        # last_sym + 1 = first symbol after the last_sym
  template = template.reshape(40, 25)
  return template

def template_gen():
    # template:
    x = [1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6,
         7, 7, 7, 7, 7, 7,
         8, 8, 8, 8,
         9, 9, 9, 9, 9, 9,
         10, 10, 10, 10, 10,
         11, 11, 11, 11, 11, 11, 11, 11,
         12, 12, 12, 12, 12, 12,
         13, 13, 13, 13, 13,
         14, 14, 14, 14, 14, 14,
         15, 15, 15, 15,
         16, 16, 16, 16, 16,
         17, 17, 17, 17, 17, 17,
         18, 18, 18, 18, 18, 18,
         19, 19, 19, 19, 19, 19, 19,
         20, 20, 20, 20, 20, 20, 20,
         21, 21, 21, 21, 21, 21, 21,
         22, 22, 22, 22, 22, 22, 22, 22,
         23, 23, 23, 23,
         24, 24, 24, 24, 24,
         25, 25, 25, 25, 25, 25, 25, 25,
         26, 26, 26, 26, 26, 26, 26, 26,
         27, 27, 27, 27, 27, 27,
         28, 28, 28,
         29, 29, 29, 29, 29, 29, 29, 29, 29,
         30, 30, 30, 30,
         31, 31, 31, 31, 31, 31, 31, 31,
         32, 32, 32, 32, 32, 32, 32, 32,
         33, 33, 33, 33, 33, 33, 33,
         34, 34, 34, 34,
         35, 35, 35, 35, 35,
         36, 36, 36,
         37, 37, 37, 37, 37, 37, 37,
         38, 38, 38, 38, 38, 38, 38,
         39, 39, 39, 39, 39, 39, 39, 39]  # rows
    y = [1, 2, 5, 9, 12, 19,
         3, 6, 10, 13, 16, 23,
         0, 1, 3, 4, 5, 9, 18, 20, 23, 24,
         1, 8, 11, 15, 18, 21,
         0, 4, 8, 12, 16, 19, 20, 24,
         1, 8, 11, 15, 19,
         2, 6, 12, 14, 17, 22,
         2, 3, 6, 11,
         3, 8, 9, 13, 18, 23,
         0, 7, 10, 14, 24,
         2, 4, 5, 9, 13, 15, 17, 18,
         2, 5, 10, 14, 16, 22,
         8, 12, 16, 20, 22,
         3, 6, 11, 21, 23, 24,
         3, 10, 17, 23,
         5, 9, 11, 14, 18,
         1, 6, 10, 13, 16, 21,
         1, 6, 12, 17, 21, 22,
         0, 4, 7, 15, 19, 20, 24,
         1, 6, 9, 13, 16, 17, 21,
         0, 4, 13, 14, 17, 20, 24,
         2, 8, 11, 12, 14, 16, 20, 23,
         1, 6, 15, 19,
         1, 7, 11, 18, 21,
         3, 7, 10, 14, 15, 19, 23, 24,
         0, 4, 5, 9, 15, 19, 21, 22,
         2, 5, 6, 9, 17, 22,
         7, 13, 19,
         0, 4, 7, 10, 14, 15, 17, 20, 24,
         3, 5, 16, 23,
         0, 8, 11, 14, 18, 20, 21, 22,
         3, 4, 7, 10, 12, 17, 20, 23,
         2, 7, 8, 12, 16, 18, 22,
         6, 10, 14, 21,
         1, 11, 12, 18, 21,
         2, 8, 12,
         3, 5, 7, 11, 16, 19, 23,
         0, 4, 7, 13, 18, 22, 24,
         2, 8, 9, 10, 13, 15, 19, 22]  # columns
    def listmaker(n):
        listofones = [1] * n
        return listofones
    value = listmaker(241)  # list of ones for each true positive position
    row = np.array(x)
    col = np.array(y)
    data = np.array(value)
    template = csr_matrix((data, (row, col)), shape=(40, 25)).toarray()
    return template

def perf_anal_pat(excel_path):
  #patient performance from ground truth:
  stop = pd.read_excel(excel_path, sheet_name='T2B').to_numpy()
  if np.isnan(stop[12][1]):                                                                                             # in case s.o. is faster than 10min
      if np.isnan(stop[11][1]):                                                                                         # in case s.o. is faster than 9min
          stop = (stop[10][1], stop[10][2])
      elif np.isnan(stop[12][1]):
          stop = (stop[11][1], stop[11][2])
  else:
      stop = (stop[12][1], stop[12][2])
  stop_indx = (stop[0]-1)*25 + (stop[1]-1)
  pat = pd.read_excel(excel_path, sheet_name='template', header= None)
  pat = pat.to_numpy()
  train_row = np.zeros(25, dtype=int)
  pat = np.vstack((train_row, pat))
  pat_template = template_gen()
  pat_template[np.where(pat == 'a')] = 0
  pat_template[np.where(pat == 'f')] = 1
  pat = pat_template.flatten()[:(stop_indx + 1)]
  template = template_gen().flatten()[:(stop_indx + 1)]
  #quantitative analysis:
  TP = np.sum(np.logical_and(pat == 1, template == 1))
  TN = np.sum(np.logical_and(pat == 0, template == 0))
  FP = np.sum(np.logical_and(pat == 1, template == 0))
  FN = np.sum(np.logical_and(pat == 0, template == 1))
  b_ACC = 0.5*((TP/(TP+FN))+(TN/(TN+FP)))
  F1 = 2*((TP/(TP+FP)) * (TP/(TP+FN))) / ((TP/(TP+FP)) + (TP/(TP+FN)))
  SEN = TP/(TP+FN)
  SPE = TN/(TN+FP)
  FN_loc = np.logical_and(pat == 0, template == 1)
  FN_indx = np.transpose(np.nonzero(FN_loc == True))
  FP_loc = np.logical_and(pat == 1, template == 0)
  FP_indx = np.transpose(np.nonzero(FP_loc == True))
  #create dataframe:
  def df_creator(excel_path):
    data = {'TP': [TP],
            'TN': [TN],
            'FP': [FP],
            'FN': [FN],
            'b-ACC': [b_ACC],
            'F1': [F1],
            'SEN': [SEN],
            'SPE': [SPE]}
    df = pd.DataFrame(data)
    excel_name = os.path.basename(excel_path)
    excel_name = os.path.splitext(excel_name)[0]
    df.index = [excel_name]
    return df
  df = df_creator(excel_path)
  return df

def perf_anal_script(excel_path, model):
  #performance analysis:
  jpeg_path = os.path.splitext(excel_path)[0] + ".jpeg"
  #last symbol from excel sheet:
  stop = pd.read_excel(excel_path, sheet_name='T2B').to_numpy()
  if np.isnan(stop[12][1]):                                                                                             # in case s.o. is faster than 10min
      if np.isnan(stop[11][1]):                                                                                         # in case s.o. is faster than 9min
          stop = (stop[10][1], stop[10][2])
      elif np.isnan(stop[12][1]):
          stop = (stop[11][1], stop[11][2])
  else:
      stop = (stop[12][1], stop[12][2])
  T2B = T2B_import(jpeg_path)
  img = T2B[0]
  dim = T2B[1]
  row_peaks = T2B_splitter(img, dim)[0]
  clmn_peaks = T2B_splitter(img, dim)[1]
  adapt_row = row_adaptor(row_peaks)
  try:
      any(adapt_row == None)
  except:
      img = ndimage.rotate(img, 1, reshape=False)                                                                       # minimal rotation for skewed test sheet
      dim = img.shape
      row_peaks = T2B_splitter(img, dim)[0]
      clmn_peaks = T2B_splitter(img, dim)[1]
      adapt_row = row_adaptor(row_peaks)
      adapt_clmn = clmn_adaptor(clmn_peaks)
  adapt_clmn = clmn_adaptor(clmn_peaks)
  try:
      any(adapt_clmn == None)
  except:
      img = ndimage.rotate(img, 1, reshape=False)                                                                       # minimal rotation for skewed test sheet
      dim = img.shape
      row_peaks = T2B_splitter(img, dim)[0]
      clmn_peaks = T2B_splitter(img, dim)[1]
      adapt_row = row_adaptor(row_peaks)
      adapt_clmn = clmn_adaptor(clmn_peaks)
  sym_list = sym_list_loop(img, adapt_row, adapt_clmn)[0]
  mean_sym_list = sym_list_loop(img, adapt_row, adapt_clmn)[1]
  mark = model(sym_list, mean_sym_list)[0]
  mean_in_areas = model(sym_list, mean_sym_list)[1]
  #ground truth from excel sheet:
  gt = pd.read_excel(excel_path, sheet_name='template', header= None)
  gt = gt.to_numpy()
  train_row = np.zeros(25, dtype=int)
  gt = np.vstack((train_row, gt))
  template = adapt_template(stop)
  template[np.where(gt == 'a')] = 0
  template[np.where(gt == 'f')] = 1
  gt = template
  #quantitative analysis:
  TP = np.sum(np.logical_and(mark == 1, gt == 1))
  TN = np.sum(np.logical_and(mark == 0, gt == 0))
  FP = np.sum(np.logical_and(mark == 1, gt == 0))
  FN = np.sum(np.logical_and(mark == 0, gt == 1))
  b_ACC = 0.5*((TP/(TP+FN))+(TN/(TN+FP)))
  F1 = 2*((TP/(TP+FP)) * (TP/(TP+FN))) / ((TP/(TP+FP)) + (TP/(TP+FN)))
  SEN = TP/(TP+FN)
  SPE = TN/(TN+FP)
  FN_loc = np.logical_and(mark == 0, gt == 1)
  FN_indx = np.transpose(np.nonzero(FN_loc == True))
  FP_loc = np.logical_and(mark == 1, gt == 0)
  FP_indx = np.transpose(np.nonzero(FP_loc == True))
  #create dataframe
  def df_creator(excel_path):
    data = {'TP': [TP],
            'TN': [TN],
            'FP': [FP],
            'FN': [FN],
            'b-ACC': [b_ACC],
            'F1': [F1],
            'SEN': [SEN],
            'SPE': [SPE]}
    df = pd.DataFrame(data)
    excel_name = os.path.basename(excel_path)
    excel_name = os.path.splitext(excel_name)[0]
    df.index = [excel_name]
    return df
  df = df_creator(excel_path)
  return df

def perf_anal_clin(excel_path):
  # clinical evaluation from excel sheet:
  stop = pd.read_excel(excel_path, sheet_name='T2B').to_numpy()
  if np.isnan(stop[12][1]):                                                                                             # in case s.o. is faster than 10min
      if np.isnan(stop[11][1]):                                                                                         # in case s.o. is faster than 9min
          stop = (stop[10][1], stop[10][2])
      elif np.isnan(stop[12][1]):
          stop = (stop[11][1], stop[11][2])
  else:
      stop = (stop[12][1], stop[12][2])
  clin = pd.read_excel(excel_path, sheet_name='template', header= None)
  clin = clin.to_numpy()
  train_row = np.zeros(25, dtype=int)
  clin = np.vstack((train_row, clin))
  template = adapt_template(stop)
  template[np.where(clin == 'a')] = 0
  template[np.where(clin == 'f')] = 1
  clin = template
  #ground truth from excel sheet:
  gt_path = os.path.splitext(excel_path)[0] + ".xlsm"
  stop = pd.read_excel(gt_path, sheet_name='T2B').to_numpy()
  if np.isnan(stop[12][1]):                                                                                             # in case s.o. is faster than 10min
      if np.isnan(stop[11][1]):                                                                                         # in case s.o. is faster than 9min
          stop = (stop[10][1], stop[10][2])
      elif np.isnan(stop[12][1]):
          stop = (stop[11][1], stop[11][2])
  else:
      stop = (stop[12][1], stop[12][2])
  gt = pd.read_excel(gt_path, sheet_name='template', header= None)
  gt = gt.to_numpy()
  gt = np.vstack((train_row, gt))
  gt_template = adapt_template(stop)
  gt_template[np.where(gt == 'a')] = 0
  gt_template[np.where(gt == 'f')] = 1
  gt = gt_template
  #quantitative analysis:
  TP = np.sum(np.logical_and(clin == 1, gt == 1))
  TN = np.sum(np.logical_and(clin == 0, gt == 0))
  FP = np.sum(np.logical_and(clin == 1, gt == 0))
  FN = np.sum(np.logical_and(clin == 0, gt == 1))
  b_ACC = 0.5*((TP/(TP+FN))+(TN/(TN+FP)))
  F1 = 2*((TP/(TP+FP)) * (TP/(TP+FN))) / ((TP/(TP+FP)) + (TP/(TP+FN)))
  SEN = TP/(TP+FN)
  SPE = TN/(TN+FP)
  FN_loc = np.logical_and(clin == 0, gt == 1)
  FN_indx = np.transpose(np.nonzero(FN_loc == True))
  FP_loc = np.logical_and(clin == 1, gt == 0)
  FP_indx = np.transpose(np.nonzero(FP_loc == True))
  #create dataframe:
  def df_creator(excel_path):
    data = {'TP': [TP],
            'TN': [TN],
            'FP': [FP],
            'FN': [FN],
            'b-ACC': [b_ACC],
            'F1': [F1],
            'SEN': [SEN],
            'SPE': [SPE]}
    df = pd.DataFrame(data)
    excel_name = os.path.basename(excel_path)
    excel_name = os.path.splitext(excel_name)[0]
    df.index = [excel_name]
    return df
  df = df_creator(excel_path)
  return df

excel_paths = search_for_xlsm_paths()                                                                                   #for script performance
#T2B_evaluator(excel_paths, model_strict_thresh)

## patient performance:

pat_performance = []
for indx in range(len(excel_paths)):
    excel_path = excel_paths[indx]
    pa = perf_anal_pat(excel_path)
    pat_performance.append(pa)

pat_performance = pd.concat(pat_performance)
avg_perf = pd.DataFrame(pat_performance[['b-ACC', 'F1', 'SEN', 'SPE']].mean(axis = 0)).T
avg_perf.index = ['average performance']
pat_perf = pd.concat([pat_performance, avg_perf], axis = 0).fillna(0)
pat_perf = pat_perf.astype({'TP': int, 'TN': int, 'FP': int, 'FN': int})
pat_perf.iloc[-1,[0, 1, 2, 3]] = ''
macro_perf = pd.DataFrame({
   'TP': [pat_performance.sum()[0].astype(int)],
   'TN': [pat_performance.sum()[1].astype(int)],
   'FP': [pat_performance.sum()[2].astype(int)],
   'FN': [pat_performance.sum()[3].astype(int)],
   'b-ACC': [0.5*((pat_performance.sum()[0].astype(int)/(pat_performance.sum()[0].astype(int)+pat_performance.sum()[3].astype(int)))+(pat_performance.sum()[1].astype(int)/(pat_performance.sum()[1].astype(int)+pat_performance.sum()[2].astype(int))))],
   'F1': [2*((pat_performance.sum()[0].astype(int)/(pat_performance.sum()[0].astype(int)+pat_performance.sum()[2].astype(int)))*(pat_performance.sum()[0].astype(int)/(pat_performance.sum()[0].astype(int)+pat_performance.sum()[3].astype(int))))/((pat_performance.sum()[0].astype(int)/(pat_performance.sum()[0].astype(int)+pat_performance.sum()[2].astype(int)))+(pat_performance.sum()[0].astype(int)/(pat_performance.sum()[0].astype(int)+pat_performance.sum()[3].astype(int))))],
   'SEN': [pat_performance.sum()[0].astype(int)/(pat_performance.sum()[0].astype(int)+pat_performance.sum()[3].astype(int))],
   'SPE': [pat_performance.sum()[1].astype(int)/(pat_performance.sum()[1].astype(int)+pat_performance.sum()[2].astype(int))]})
macro_perf.index = ['macro performance']
pat_perf = pd.concat([pat_perf, macro_perf], axis=0)
pat_perf.to_excel('Patient performance.xlsx')

### script performance:
# To run the performance analysis of the models use the files in folder 'Performance analysis - Script,
# consisting of .xlsm files (ground truth of patient's) and test sheet images (.jpeg). Both types of files need to be in
# the same folder. The tkinter window will prompt you to select the ground truth files and the script will automatically
# use the corresponding images to run the classification models on, as long as they are in the same folder and have the
# same name (T2B_date_...). At the end of the code a new Excel file will be generated with the calculated performance
# analysis of the models.

## Model with inside OR outside criteria:

script_performance = []
for indx in range(len(excel_paths)):
  excel_path = excel_paths[indx]
  pa = perf_anal_script(excel_path, model_in_OR_out)
  script_performance.append(pa)

script_performance = pd.concat(script_performance)
avg_perf = pd.DataFrame(script_performance[['b-ACC', 'F1', 'SEN', 'SPE']].mean(axis = 0)).T
avg_perf.index = ['average performance']
script_perf = pd.concat([script_performance, avg_perf], axis = 0).fillna(0)
script_perf = script_perf.astype({'TP': int, 'TN': int, 'FP': int, 'FN': int})
script_perf.iloc[-1,[0, 1, 2, 3]] = ''
macro_perf = pd.DataFrame({
    'TP': [script_performance.sum()[0].astype(int)],
    'TN': [script_performance.sum()[1].astype(int)],
    'FP': [script_performance.sum()[2].astype(int)],
    'FN': [script_performance.sum()[3].astype(int)],
    'b-ACC': [0.5*((script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int)))+(script_performance.sum()[1].astype(int)/(script_performance.sum()[1].astype(int)+script_performance.sum()[2].astype(int))))],
    'F1': [2*((script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[2].astype(int)))*(script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int))))/((script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[2].astype(int)))+(script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int))))],
    'SEN': [script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int))],
    'SPE': [script_performance.sum()[1].astype(int)/(script_performance.sum()[1].astype(int)+script_performance.sum()[2].astype(int))]})
macro_perf.index = ['macro performance']
Model_in_or_out_perf = pd.concat([script_perf, macro_perf], axis=0)
Model_in_or_out_perf.to_excel('Performance analysis Model in_or_out criteria.xlsx')

## Model with inside AND outside criteria:

script_performance = []
for indx in range(len(excel_paths)):
  excel_path = excel_paths[indx]
  pa = perf_anal_script(excel_path, model_in_AND_out)
  script_performance.append(pa)

script_performance = pd.concat(script_performance)
avg_perf = pd.DataFrame(script_performance[['b-ACC', 'F1', 'SEN', 'SPE']].mean(axis = 0)).T
avg_perf.index = ['average performance']
script_perf = pd.concat([script_performance, avg_perf], axis = 0).fillna(0)
script_perf = script_perf.astype({'TP': int, 'TN': int, 'FP': int, 'FN': int})
script_perf.iloc[-1,[0, 1, 2, 3]] = ''
macro_perf = pd.DataFrame({
    'TP': [script_performance.sum()[0].astype(int)],
    'TN': [script_performance.sum()[1].astype(int)],
    'FP': [script_performance.sum()[2].astype(int)],
    'FN': [script_performance.sum()[3].astype(int)],
    'b-ACC': [0.5*((script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int)))+(script_performance.sum()[1].astype(int)/(script_performance.sum()[1].astype(int)+script_performance.sum()[2].astype(int))))],
    'F1': [2*((script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[2].astype(int)))*(script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int))))/((script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[2].astype(int)))+(script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int))))],
    'SEN': [script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int))],
    'SPE': [script_performance.sum()[1].astype(int)/(script_performance.sum()[1].astype(int)+script_performance.sum()[2].astype(int))]})
macro_perf.index = ['macro performance']
Model_in_and_out_perf = pd.concat([script_perf, macro_perf], axis=0)
Model_in_and_out_perf.to_excel('Performance analysis Model in_and_out criteria.xlsx')

## Model with strict threshold for non-target symbols:

script_performance = []
for indx in range(len(excel_paths)):
  excel_path = excel_paths[indx]
  pa = perf_anal_script(excel_path, model_strict_thresh)
  script_performance.append(pa)

script_performance = pd.concat(script_performance)
avg_perf = pd.DataFrame(script_performance[['b-ACC', 'F1', 'SEN', 'SPE']].mean(axis = 0)).T
avg_perf.index = ['average performance']
script_perf = pd.concat([script_performance, avg_perf], axis = 0).fillna(0)
script_perf = script_perf.astype({'TP': int, 'TN': int, 'FP': int, 'FN': int})
script_perf.iloc[-1,[0, 1, 2, 3]] = ''
macro_perf = pd.DataFrame({
    'TP': [script_performance.sum()[0].astype(int)],
    'TN': [script_performance.sum()[1].astype(int)],
    'FP': [script_performance.sum()[2].astype(int)],
    'FN': [script_performance.sum()[3].astype(int)],
    'b-ACC': [0.5*((script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int)))+(script_performance.sum()[1].astype(int)/(script_performance.sum()[1].astype(int)+script_performance.sum()[2].astype(int))))],
    'F1': [2*((script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[2].astype(int)))*(script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int))))/((script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[2].astype(int)))+(script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int))))],
    'SEN': [script_performance.sum()[0].astype(int)/(script_performance.sum()[0].astype(int)+script_performance.sum()[3].astype(int))],
    'SPE': [script_performance.sum()[1].astype(int)/(script_performance.sum()[1].astype(int)+script_performance.sum()[2].astype(int))]})
macro_perf.index = ['macro performance']
Model_strict_threshold_perf = pd.concat([script_perf, macro_perf], axis=0)
Model_strict_threshold_perf.to_excel('Performance analysis Model with strict threshold.xlsx')

## clinical performance:
# To run the performance analysis of the clinicians use the files in folder 'Performance Analysis - Clinicians'.
# Both .xlsx (clinician's evaluation) and .xlsm (ground truth of patient's) Excel files need to be in the same folder.
# The code is designed to access both types of Excel files by name.
# The tkinter window will prompt you to select the .xlsx files and the script will automatically select
# the corresponding .xlsm files as long as they are in the same folder and have the same name (T2B_date_...).
# At the end of the code a new Excel file will be generated with the calculated performance analysis.

clin_paths = search_for_xlsx_paths()                                                                                    #for clinical performance

clin_performance = []
for indx in range(len(clin_paths)):
    clin_path = clin_paths[indx]
    pa = perf_anal_clin(clin_path)
    clin_performance.append(pa)

clin_performance = pd.concat(clin_performance)
avg_perf = pd.DataFrame(clin_performance[['b-ACC', 'F1', 'SEN', 'SPE']].mean(axis = 0)).T
avg_perf.index = ['average performance']
clin_perf = pd.concat([clin_performance, avg_perf], axis = 0).fillna(0)
clin_perf = clin_perf.astype({'TP': int, 'TN': int, 'FP': int, 'FN': int})
clin_perf.iloc[-1,[0, 1, 2, 3]] = ''
macro_perf = pd.DataFrame({
   'TP': [clin_performance.sum()[0].astype(int)],
   'TN': [clin_performance.sum()[1].astype(int)],
   'FP': [clin_performance.sum()[2].astype(int)],
   'FN': [clin_performance.sum()[3].astype(int)],
   'b-ACC': [0.5*((clin_performance.sum()[0].astype(int)/(clin_performance.sum()[0].astype(int)+clin_performance.sum()[3].astype(int)))+(clin_performance.sum()[1].astype(int)/(clin_performance.sum()[1].astype(int)+clin_performance.sum()[2].astype(int))))],
   'F1': [2*((clin_performance.sum()[0].astype(int)/(clin_performance.sum()[0].astype(int)+clin_performance.sum()[2].astype(int)))*(clin_performance.sum()[0].astype(int)/(clin_performance.sum()[0].astype(int)+clin_performance.sum()[3].astype(int))))/((clin_performance.sum()[0].astype(int)/(clin_performance.sum()[0].astype(int)+clin_performance.sum()[2].astype(int)))+(clin_performance.sum()[0].astype(int)/(clin_performance.sum()[0].astype(int)+clin_performance.sum()[3].astype(int))))],
   'SEN': [clin_performance.sum()[0].astype(int)/(clin_performance.sum()[0].astype(int)+clin_performance.sum()[3].astype(int))],
   'SPE': [clin_performance.sum()[1].astype(int)/(clin_performance.sum()[1].astype(int)+clin_performance.sum()[2].astype(int))]})
macro_perf.index = ['macro performance']
clin_perf = pd.concat([clin_perf, macro_perf], axis=0)
clin_perf.to_excel('clinical performance.xlsx')

### RunTime loop:
import time

def evaluation(mark, template):
  #we find the predicted and true labels that are assigned to some specific class
  #then we use the "AND" operator to combine the results of the two label vectors
  #into a single binary vector
  #then we sum over the binary vector to count how many incidences there are
  # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
  TP = np.sum(np.logical_and(mark == 1, template == 1))
  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
  TN = np.sum(np.logical_and(mark == 0, template == 0))
  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
  FP = np.sum(np.logical_and(mark == 1, template == 0))
  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
  FN = np.sum(np.logical_and(mark == 0, template == 1))
  # print(TP)
  # print(TN)
  # print(FN)
  # print(FP)
  # Accuracy (ACC): accuracy is the fraction of predictions our model got right
  ACC = (TP+TN)/(TP+TN+FP+FN)
  # print(ACC)
  return TP, TN, FP, FN, ACC

def visual(img, mark, template, adapt_row, adapt_clmn):
  FN_loc = np.logical_and(mark == 0, template == 1)
  FN_indx = np.transpose(np.nonzero(FN_loc == True))
  FP_loc = np.logical_and(mark == 1, template == 0)
  FP_indx = np.transpose(np.nonzero(FP_loc == True))
  #visualization:
  plt.rcParams["figure.figsize"] = (30, 16)
  plt.axis('off')
  plt.imshow(img, cmap='gray')
  for w in range(len(FN_indx)):
    plt.hlines((adapt_row[FN_indx[w, 0]], adapt_row[(FN_indx[w, 0] + 1)]), adapt_clmn[FN_indx[w, 1]], adapt_clmn[(FN_indx[w, 1] + 1)], colors = ("orange"), linewidth = 1)
    plt.vlines((adapt_clmn[FN_indx[w, 1]], adapt_clmn[(FN_indx[w, 1] + 1)]), adapt_row[FN_indx[w, 0]], adapt_row[(FN_indx[w, 0] + 1)], colors = ("orange"), linewidth = 1)
  for v in range(len(FP_indx)):
    plt.hlines((adapt_row[FP_indx[v, 0]], adapt_row[(FP_indx[v, 0] + 1)]), adapt_clmn[FP_indx[v, 1]], adapt_clmn[(FP_indx[v, 1] + 1)], colors = ("r"), linewidth = 1)
    plt.vlines((adapt_clmn[FP_indx[v, 1]], adapt_clmn[(FP_indx[v, 1] + 1)]), adapt_row[FP_indx[v, 0]], adapt_row[(FP_indx[v, 0] + 1)], colors = ("r"), linewidth = 1)
  vis = plt.show()
  return vis, FN_indx, FP_indx

def FNFP_visual(img, FN_indx, FP_indx, adapt_row, adapt_clmn, criteria):
  for w in range(len(FN_indx)):
    FN_img = img[adapt_row[FN_indx[w, 0]]:adapt_row[(FN_indx[w, 0] + 1)], adapt_clmn[FN_indx[w, 1]]:adapt_clmn[(FN_indx[w, 1] + 1)]]
    cv.imshow(FN_img)
    print(FN_indx[w])
    if criteria[FN_indx[w][0],FN_indx[w][1]] == 0:
      print('inside')
    else:
      print('outside')
  for v in range(len(FP_indx)):
    FP_img = img[adapt_row[FP_indx[v, 0]]:adapt_row[(FP_indx[v, 0] + 1)], adapt_clmn[FP_indx[v, 1]]:adapt_clmn[(FP_indx[v, 1] + 1)]]
    cv.imshow(FP_img)
    print(FP_indx[v])
    if criteria[FP_indx[v][0],FP_indx[v][1]] == 0:
      print('inside')
    else:
      print('outside')

def T2B_evaluator(excel_path, model):
  # patient performance from ground truth:
  stop = pd.read_excel(excel_path, sheet_name='T2B').to_numpy()
  if np.isnan(stop[12][1]):  # in case s.o. is faster than 10min
    if np.isnan(stop[11][1]):  # in case s.o. is faster than 9min
        stop = (stop[10][1], stop[10][2])
    elif np.isnan(stop[12][1]):
        stop = (stop[11][1], stop[11][2])
  else:
    stop = (stop[12][1], stop[12][2])
  stop_indx = (stop[0] - 1) * 25 + (stop[1] - 1)
  jpeg_path = os.path.splitext(excel_path)[0] + ".jpeg"
  T2B = T2B_import(jpeg_path)
  img = T2B[0]
  dim = T2B[1]
  row_peaks = T2B_splitter(img, dim)[0]
  clmn_peaks = T2B_splitter(img, dim)[1]
  adapt_row = row_adaptor(row_peaks)
  try:
      any(adapt_row == None)
  except:
      img = ndimage.rotate(img, 1, reshape=False)                                                                       # minimal rotation for skewed test sheet
      dim = img.shape
      row_peaks = T2B_splitter(img, dim)[0]
      clmn_peaks = T2B_splitter(img, dim)[1]
      adapt_row = row_adaptor(row_peaks)
      adapt_clmn = clmn_adaptor(clmn_peaks)
  adapt_clmn = clmn_adaptor(clmn_peaks)
  try:
      any(adapt_clmn == None)
  except:
      img = ndimage.rotate(img, 1, reshape=False)                                                                       # minimal rotation for skewed test sheet
      dim = img.shape
      row_peaks = T2B_splitter(img, dim)[0]
      clmn_peaks = T2B_splitter(img, dim)[1]
      adapt_row = row_adaptor(row_peaks)
      adapt_clmn = clmn_adaptor(clmn_peaks)
  sym_list = sym_list_loop(img, adapt_row, adapt_clmn)[0]
  mean_sym_list = sym_list_loop(img, adapt_row, adapt_clmn)[1]
  mark = model(sym_list, mean_sym_list)[0]
  mark = mark.flatten()[:(stop_indx + 1)]
  template = template_gen().flatten()[:(stop_indx + 1)]
  eval = evaluation(mark, template)
  #fill_template = template_gen().flatten()[(stop_indx + 1):]
  #mark = np.append(mark, fill_template)
  #mark = mark.reshape(40, 25)
  #template = template_gen()
  #vis = visual(img, mark, template, adapt_row, adapt_clmn)[0]
  return eval

#st = start time
st = time.time()
for indx in range(len(excel_paths)):
    excel_path = excel_paths[indx]
    T2B_evaluator(excel_path, model_strict_thresh)
#et = end time
et = time.time()
# get the execution time
elapsed_time = et - st
avg_time = (et - st) / 23
print('Execution time:', elapsed_time, 'seconds')
print('Execution time:', avg_time, 'seconds')
