**Exceptions, that need consideration in the code:**

split_adapt function:

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

symbol_splitter function:

adada

symbol number 412, row 17/ clmn 13 doesn't have two row_vals (26, x)

symbol number 690, row 28/ clmn 16 doesn't have two row_vals (x, 46)

symbol number 712, row 29/ clmn 13 doens't have two row_vals (25, x)

symbol number 920, row 37/ clmn 13 doesn't have two clmn_vals (x, 48)

for symbols number 786, 906, 920:

\- symbols having more than two minima

for symbols number 240, 326, 558:

\- symbols missing one minima, but having two minima
