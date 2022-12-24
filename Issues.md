**Issues, that need consideration in the code:**

**split_adapt function:**

In some cases, the *split_adapt* function output too many or too few *rows*. *If* and *while* terms did not correct the output*,* because for some reason every input (here *row_peaks* and *clmn_peaks*) were taken into consideration in the *if* and *while* terms.

**Solution:**

Divide function into two separate ones with different names. So, the *while* term only takes one input into account for the output.

```
def row_adaptor(x):
  for std in range(3, 6):               #std = 3, 4, 5 work the best, is there alternative to np.floor(np.std(np.diff(x)))??
    distance = np.diff(x)
    median = int(np.median(np.diff(x))) 
    up_bound = median + std             
    low_bound = median - std
    inlier = np.asarray(np.where((distance > low_bound) & (distance < up_bound))).flatten(order = 'C')
    index = [inlier[0], (inlier[-1] + 2)] #np.diff output one element less than original array, and + 2 to accomodate this difference
    frame = x[index[0]:index[-1]]
    split = np.insert(frame, 0, (frame[0] - median))
    split = np.append(split, (frame[-1] + median))
    print(len(split))
    while len(split) == 41:
      return split

def clmn_adaptor(x):
  for std in range(3, 6):               #std = 3, 4, 5 work the best, is there alternative to np.floor(np.std(np.diff(x)))??
    distance = np.diff(x)
    median = int(np.median(np.diff(x))) 
    up_bound = median + std             
    low_bound = median - std
    inlier = np.asarray(np.where((distance > low_bound) & (distance < up_bound))).flatten(order = 'C')
    index = [inlier[0], (inlier[-1] + 2)] #np.diff output one element less than original array, and + 2 to accomodate this difference
    frame = x[index[0]:index[-1]]
    split = np.insert(frame, 0, (frame[0] - median))
    split = np.append(split, (frame[-1] + median))
    print(len(split))
    while len(split) == 26:
      return split
```

**symbol_splitter function:**

While checking the *symbol_splitter* function on the unused test sheet, some errors arose. In some cases, symbols only had one of two expected *row_vals* or *clmn_vals*:

-   symbol number 412, [row 17/ clmn 13] doesn't have two row_vals (26, x)
-   symbol number 690, [row 28/ clmn 16] doesn't have two row_vals (x, 46)
-   symbol number 712, [row 29/ clmn 13] doens't have two row_vals (25, x)
-   symbol number 920, [row 37/ clmn 13] doesn't have two clmn_vals (x, 48)

**Solution:** Add if term to account for less than two minima.

```
if len(row_vals) == 1:          #symbols with only one minima
    if row_vals[0] < 30:
      row_vals = np.append(row_vals, (row_vals[0] + 21))		#symbol number 412 (row 17/ clmn 13) and 712 (row 29/ clmn 13) don't have second minima!
    elif row_vals[0] > 30:
      row_vals = np.insert(row_vals, 0, (row_vals[0] - 21))	#symbol number 690 (row 28/ clmn 16) doesn't have first minima!
```

In other cases, symbols had more than two minima:

-   symbols number 786, 906, 920

**Solution:** Add if term to take only the first and last minima.

```
if len(row_vals) != 2:          #symbols having more than two minima
    row_vals = row_vals[0], row_vals[-1]
```

Finally, some symbols had two minima, but the distances between both minima were too short:

-   symbols number 240, 326, 558

**Solution:** Add if term to check for distances and adapt minima accordingly.

```
if np.diff(row_vals) < 15:
    if row_vals[0] > 30:
      row_vals = np.insert(row_vals, 0, (row_vals[1] - 21))
    elif row_vals[1] < 40:        #symbols missing one minima, but having two minima
      row_vals = np.append(row_vals, (row_vals[0] + 21))
```

**in_symbol mark recognition:**

While checking, whether the mean grayscale value of the area *in_symbol* is indicative of a mark by a patient, some test sheets showed differences:

-   On many test sheets symbol [12, 11] was mistaken as FP. There seems to have been an irregularity on the scanning glass.
-   Test sheet “T2B_10.10.2022.1.jpeg” had another FP. The black border of the symbol was still visible in *in_symbol*. This time the +- 3 on each side of *row_vals* and *clmn_vals* was too little and had to be upped to +- 4. (Checking whether the results differ compared to +-3)
-   Test sheet “T2B_10.10.2022.2.jpeg” has a lot of mistaken omissions (got to check in depth why)
-   Test sheet “T2B_10.26.2022” has two symbols [30:18], [32:15] mistaken as unmarked
-   Test sheet “T2B_11.03.2022” has one symbol [22:14] mistaken as unmarked
-   
-   
