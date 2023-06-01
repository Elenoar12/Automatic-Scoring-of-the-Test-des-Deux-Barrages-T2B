import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import statsmodels.api as sm
from scipy.signal import argrelmax, argrelmin
from scipy.sparse import csr_matrix
import pandas as pd

root = tk.Tk()
root.withdraw()                                                                                                         #use to hide tkinter window
def search_for_file_path():
    filepath = filedialog.askopenfilename(parent = root, title = 'Please select a file', filetypes = (("JPEG files","*.jpeg"),("all files","*.*")))
    return filepath
filepath = search_for_file_path()
jpeg_path = filepath

img = cv.imread(jpeg_path, cv.IMREAD_GRAYSCALE)
dim = img.shape
mean_row = np.mean(img, axis=1)                                                                                         #axis = 1 means working along the row
mean_clmn = np.mean(img, axis=0)                                                                                        #axis = 0 means along the column

#smoothing:

x = np.array(range(dim[0]))
y = np.array(range(dim[-1]))
lowess = sm.nonparametric.lowess
smth_row = lowess(mean_row, range(len(x)), return_sorted = False, frac = 0.027)                                         #frac parameter for clmn/row are opposite because x/y are swapped when rotated 90 degree
row_peaks = argrelmax(smth_row)

smth_clmn = lowess(mean_clmn, range(len(y)), return_sorted = False, frac = 0.04)
clmn_peaks = argrelmax(smth_clmn)

#splits:

row_peaks = np.asarray(row_peaks)                                                                                       #original row and column vals saved as tuple[ndarray]
clmn_peaks = np.asarray(clmn_peaks)
row_peaks = row_peaks.flatten(order='C')
clmn_peaks = clmn_peaks.flatten(order='C')

#adapt boundaries:

def row_adaptor(row_peaks, exp_range = (300, 3000)):
  for std in range(3, 6):               #std = 3, 4, 5 work the best, is there alternative to np.floor(np.std(np.diff(x)))??
    distance = np.diff(row_peaks)
    median = int(np.median(np.diff(row_peaks)))
    up_bound = median + std
    low_bound = median - std
    row_peaks = row_peaks[np.where(np.logical_and(row_peaks > exp_range[0],
                                                    row_peaks < exp_range[1]))] #filters clmn_peak for expected range (hardcoded!!!)
    too_close = np.array(np.where(np.logical_or(np.diff(row_peaks) < low_bound,
                                      np.diff(row_peaks) > up_bound))).flatten()
    dis_too_close = np.diff(too_close).flatten()
    indx_too_close = (np.array(np.where(dis_too_close == 1)).
                      flatten() + 1).astype(int)                                #+1 returns indices of indices of peaks that are too close to each other
    indx_too_close = too_close[indx_too_close]
    frame = np.delete(row_peaks, indx_too_close)
    split = np.insert(frame, 0, (frame[0] - median))
    split = np.append(split, (frame[-1] + median))
    while len(split) == 41:
      return split

def clmn_adaptor(clmn_peaks, exp_range = (300, 2000)):
  for std in range(3, 6):               #std = 3, 4, 5 work the best, is there alternative to np.floor(np.std(np.diff(x)))??
    distance = np.diff(clmn_peaks)
    median = int(np.median(np.diff(clmn_peaks)))
    up_bound = median + std
    low_bound = median - std
    clmn_peaks = clmn_peaks[np.where(np.logical_and(clmn_peaks > exp_range[0],
                                                    clmn_peaks < exp_range[1]))] #filters clmn_peak for expected range (hardcoded!!!)
    too_close = np.array(np.where(np.logical_or(np.diff(clmn_peaks) < low_bound,
                                      np.diff(clmn_peaks) > up_bound))).flatten()
    dis_too_close = np.diff(too_close).flatten()
    indx_too_close = (np.array(np.where(dis_too_close == 1)).
                      flatten() + 1).astype(int)                                #+1 returns indices of indices of peaks that are too close to each other
    indx_too_close = too_close[indx_too_close]
    frame = np.delete(clmn_peaks, indx_too_close)
    split = np.insert(frame, 0, (frame[0] - median))
    split = np.append(split, (frame[-1] + median))
    while len(split) == 26:
      return split

adapt_row = row_adaptor(row_peaks)
adapt_clmn = clmn_adaptor(clmn_peaks)

#loop for single symbols:

clmn_range = np.array(range(len(adapt_clmn)))
row_range = np.array(range(len(adapt_row)))

sym_list = []
mean_sym_list = []
row_indx_list = []
clmn_indx_list = []
for i in row_range[1:]:                                                                                                 #von index 1 ansonsten ist i-1 oder j-1 out of bounds
  for j in clmn_range[1:]:
    symbols = img[adapt_row[i-1]:adapt_row[i], adapt_clmn[j-1]:adapt_clmn[j]]
    if i == len(row_range):
      break
      if j == len(clmn_range):
        break
    row_indx = [i-1]
    clmn_indx = [j-1]
    row_indx_list.append(row_indx)
    clmn_indx_list.append(clmn_indx)
    sym_list.append(symbols)
    mean_sym_list.append(np.mean(symbols))                                                                              #to determine amount of ink to identify corrections made by the participant

row_indx_list = [item for sublist in row_indx_list for item in sublist]                                                 #turn list of lists into one flat list
clmn_indx_list = [item for sublist in clmn_indx_list for item in sublist]                                               #turn list of lists into one flat list

#function to find row and clmn vals of single symbols:

def symbol_splitter(x):
  dim_sym = x.shape
  sym_mean_row = np.mean(x, axis=1)
  sym_mean_clmn = np.mean(x, axis=0)
  row_x = np.array(range(dim_sym[0]))
  clmn_y = np.array(range(dim_sym[-1]))
  lowess = sm.nonparametric.lowess
  sym_smth_row = lowess(sym_mean_row, range(len(row_x)), return_sorted = False, frac = 0.11)                            #adjusted from 0.15 to 0.11 because some symbols had no vals!
  sym_row_vals = np.asarray(argrelmin(sym_smth_row)).flatten()
  sym_vals = sym_smth_row[sym_row_vals]
  sym_vals_indx = np.asarray(np.where(sym_vals < 220)).flatten()
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
  sym_smth_clmn = lowess(sym_mean_clmn, range(len(clmn_y)), return_sorted = False, frac = 0.11)
  sym_clmn_vals = np.asarray(argrelmin(sym_smth_clmn)).flatten()
  sym_vals = sym_smth_clmn[sym_clmn_vals]
  sym_vals_indx = np.asarray(np.where(sym_vals < 220)).flatten()
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

#determine symbol variations and areas outside the symbol:

mean_out_areas = []
mean_mark_list = []
for p in range(len(sym_list)):
  symbol = sym_list[p]
  row_vals = symbol_splitter(symbol)[1]
  clmn_vals = symbol_splitter(symbol)[2]
  out_row = row_vals[0] - 4, row_vals[1] + 4
  out_clmn = clmn_vals[0] - 4, clmn_vals[1] + 4
  in_row = row_vals[0] + 4, row_vals[1] - 4
  in_clmn = clmn_vals[0] + 4, clmn_vals[1] - 4

  #areas outside symbols:                                                                                              # slice around out_symbol to remove excess area with possible mark or symbol overlap

  out_symbol_1 = symbol[0:out_row[0], 0:(out_clmn[0])]
  out_symbol_1 = out_symbol_1[5:, 5:]
  out_symbol_2 = symbol[0:out_row[0], out_clmn[0]:out_clmn[1]]                                                          # due to issues in later calculations change lower right edge grayscale values to white (255) see below
  out_symbol_2 = out_symbol_2[5:]
  out_symbol_3 = symbol[0:out_row[0], out_clmn[1]:-1]
  out_symbol_3 = out_symbol_3[5:, :-5]
  out_symbol_4 = symbol[out_row[0]:out_row[1], out_clmn[1]:-1]
  out_symbol_4 = out_symbol_4[:, :-5]
  out_symbol_5 = symbol[out_row[1]:-1, out_clmn[1]:-1]
  out_symbol_5 = out_symbol_5[:-5, :-5]
  out_symbol_6 = symbol[out_row[1]:-1, out_clmn[0]:out_clmn[1]]
  out_symbol_6 = out_symbol_6[:-5]
  out_symbol_7 = symbol[out_row[1]:-1, 0:out_clmn[0]]
  out_symbol_7 = out_symbol_7[:-5, 5:]
  out_symbol_8 = symbol[out_row[0]:out_row[1], 0:out_clmn[0]]
  out_symbol_8 = out_symbol_8[:, 5:]

  out_symbol = [out_symbol_1, out_symbol_2, out_symbol_3, out_symbol_4, out_symbol_5, out_symbol_6, out_symbol_7,
                out_symbol_8]

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
  sym_var = mean_out.index(min(mean_out))
  if sym_var == 0:
    del mean_out[1]
    out_symbol_2[-3:, :3] = 255
    mean_out_2 = np.mean(out_symbol_2)
    mean_out.insert(1, mean_out_2)
  elif sym_var == 2:                                                                                                    #at index 2 is symbol variations 3
    del mean_out[1]                                                                                                     #delete and replace with new mean_out
    out_symbol_2[-5:, -5:] = 255                                                                                        #change lower right edge of out_symbol to white for FP correction
    mean_out_2 = np.mean(out_symbol_2)
    mean_out.insert(1, mean_out_2)
  elif sym_var == 6:                                                                                                    #at index 6 is symbol variations 7
    del mean_out[5]                                                                                                     #delete and replace with new mean_out
    out_symbol_6[:3, :3] = 255                                                                                          #change upper right edge of out_symbol to white for FP correction
    mean_out_6 = np.mean(out_symbol_6)
    mean_out.insert(5, mean_out_6)

  del mean_out[sym_var]
  mean_out = np.floor(mean_out)
  mean_out_areas.append(mean_out)

  #areas inside symbol:

  in_symbol = symbol[in_row[0]:in_row[1], in_clmn[0]:in_clmn[1]]
  mean_in_sym = np.floor(np.mean(in_symbol))
  mean_mark_list.append(mean_in_sym)

#check within symbol border for mark:

mean_mark = np.array(mean_mark_list)
mark_in = (mean_mark < 250).astype(int)                                                                                 #np.logical_and to combine two conditions without causing ValueError

#use areas outside the symbol to check for symbols without a definite mark:

mean_out_areas = np.array(mean_out_areas)
indx_mark_out = np.array(np.where(mark_in == 0)).flatten()                                                              #index for symbols that weren't recognized as marked
mark_out = np.zeros(1000, int)

for k in indx_mark_out:
  if np.count_nonzero(mean_out_areas[k] < 250) > 1:                                                                     #whenever two or more areas outside the symbol show a mark it is considered marked
    mark_out[k] = 1
  else:
    mark_out[k] = 0

#join mark arrays:

mark = np.add(mark_out, mark_in)
mark[0:25] = 0
indx_mark_sym = np.array(np.where(mark == 1)).flatten()                                                                 #needed later for stop_indx

# to determine recognition criteria (inside/outside):

#criteria = np.zeros(1000, int)
#criteria[indx_mark_in] = 0
#criteria[indx_mark_out] = 1
#criteria = criteria.reshape(40, 25)

#check for outliers/corrections:
                                                                                                                        #uncomment for mean/std correction control
# mark_sym = np.array(mean_sym_list)[indx_mark_sym]
# mean_mark_sym = np.mean(mark_sym)
# std_mark_sym = np.std(mark_sym)
# indx_outl = np.where(mark_sym < (mean_mark_sym - 2.5*std_mark_sym))
# indx_corr = indx_mark_sym[indx_outl]
# mark[indx_corr] = 0
# print(indx_mark_sym)
# print(mean_mark_sym)
# print(std_mark_sym)
# print(indx_corr)
# outl = np.array(mean_sym_list)[(np.array(indx_corr))]
# print(outl)

mark = mark.reshape(40, 25)

# template:

x = [1, 1, 1, 1, 1, 1,                                                                                                  # rows
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
     39, 39, 39, 39, 39, 39, 39, 39]
y = [1, 2, 5, 9, 12, 19,                                                                                                # columns
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
     2, 8, 9, 10, 13, 15, 19, 22]

def listmaker(n):
  listofones = [1] * n
  return listofones

value = listmaker(241)                                                                                                  # list of ones for each true positive position
row = np.array(x)
col = np.array(y)
data = np.array(value)
template = csr_matrix((data, (row, col)), shape=(40, 25)).toarray()

# we find the predicted and true labels that are assigned to some specific class
# then we use the "AND" operator to combine the results of the two label vectors
# into a single binary vector
# then we sum over the binary vector to count how many incidences there are

# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
TP = np.sum(np.logical_and(mark == 1, template == 1))

# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(mark == 0, template == 0))

# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(mark == 1, template == 0))

# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(mark == 0, template == 1))

print(TP)
print(TN)
print(FN)
print(FP)

# Accuracy (ACC): accuracy is the fraction of predictions our model got right
ACC = (TP + TN) / (TP + TN + FP + FN)
print(ACC)

FN_loc = np.logical_and(mark == 0, template == 1)
FN_indx = np.transpose(np.nonzero(FN_loc == True))
#print(FN_indx)                                                                                                         # +1 on indx values gives you the on sheet index

FP_loc = np.logical_and(mark == 1, template == 0)
FP_indx = np.transpose(np.nonzero(FP_loc == True))
#print(FP_indx)                                                                                                         # +1 on indx values gives you the on sheet index

#last marked symbol as stop for visualization:

if indx_mark_sym[-1] != 997:
  stop = indx_mark_sym[-1]
  stop_row = stop//len(clmn_range[1:])
  stop_clmn = indx_mark_sym[-1] - len(clmn_range[1:])*stop_row
  stop = np.array((stop_row, stop_clmn))
  #print(stop)
  stop_indx = np.array(np.where(np.logical_and(FN_indx[:,0] == stop[0],
                                             FN_indx[:,1] > stop[1]))).flatten()[0]
  #print(stop_indx)
  #print(FN_indx[stop_indx])
#if last FN before last marked symbol as stop_indx, range(stop_indx) would be 0 to 31
#one number short, therefore stop_indx is at the first FN after last marked symbol (0 to 32)
#same applies for rest_indx see below
  rest_indx = np.array(np.where(np.logical_and(FN_indx[:,0] == stop[0],
                                             FN_indx[:,1] > stop[1]))).flatten()[-1]
  rest_indx = rest_indx + 1     #at index -1 for last target symbol in row, +1 for range function to encompass whole range
  #print(rest_indx)
  #print(FN_indx[rest_indx])

#visualization:

  plt.rcParams["figure.figsize"] = (30, 16)
  plt.axis('off')
  plt.imshow(img, cmap='gray')
  for w in range(len(FN_indx[:stop_indx])):
    plt.hlines((adapt_row[FN_indx[w, 0]], adapt_row[(FN_indx[w, 0] + 1)]), adapt_clmn[FN_indx[w, 1]], adapt_clmn[(FN_indx[w, 1] + 1)], colors = ("orange"), linewidth = 1)
    plt.vlines((adapt_clmn[FN_indx[w, 1]], adapt_clmn[(FN_indx[w, 1] + 1)]), adapt_row[FN_indx[w, 0]], adapt_row[(FN_indx[w, 0] + 1)], colors = ("orange"), linewidth = 1)
  for v in range(len(FP_indx)):
    plt.hlines((adapt_row[FP_indx[v, 0]], adapt_row[(FP_indx[v, 0] + 1)]), adapt_clmn[FP_indx[v, 1]], adapt_clmn[(FP_indx[v, 1] + 1)], colors = ("r"), linewidth = 1)
    plt.vlines((adapt_clmn[FP_indx[v, 1]], adapt_clmn[(FP_indx[v, 1] + 1)]), adapt_row[FP_indx[v, 0]], adapt_row[(FP_indx[v, 0] + 1)], colors = ("r"), linewidth = 1)
  for u in range(stop_indx, rest_indx):
    plt.hlines((adapt_row[FN_indx[u, 0]], adapt_row[(FN_indx[u, 0] + 1)]), adapt_clmn[FN_indx[u, 1]], adapt_clmn[(FN_indx[u, 1] + 1)], colors = ("green"), linewidth = 1)
    plt.vlines((adapt_clmn[FN_indx[u, 1]], adapt_clmn[(FN_indx[u, 1] + 1)]), adapt_row[FN_indx[u, 0]], adapt_row[(FN_indx[u, 0] + 1)], colors = ("green"), linewidth = 1)

else:
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
plt.show()

