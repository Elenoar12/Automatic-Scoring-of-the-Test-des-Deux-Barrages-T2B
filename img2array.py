import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.signal import argrelmax, argrelmin

img = cv.imread(r"C:\Users\hanst\PycharmProjects\T2Bproject\pdf2array\T2B.jpeg", cv.IMREAD_GRAYSCALE)
print(img)

mean_row = np.mean(img, axis=1) #axis = 1 means working along the row
mean_column = np.mean(img, axis=0)    #axis = 0 means along the column
print(mean_column)
print(mean_row)

plt.rcParams["figure.figsize"] = (10,6)
x = np.array(range(2339))
default_x_ticks = range(len(x))
#plt.plot(default_x_ticks, mean_row)                #for visualization
#plt.xticks(default_x_ticks, x)
#plt.show()

y = np.array(range(3307))
default_x_ticks = range(len(y))
#plt.plot(default_x_ticks, mean_column)     #for visualization
#plt.xticks(default_x_ticks, y)
#plt.show()

#trial import, convert, rotate and save

pdf_path = r"C:\Users\hanst\PycharmProjects\T2Bproject\pdf2array\markT2B.pdf"
from pdf2image import convert_from_path #installing poppler package necessary
markT2B_image = convert_from_path(pdf_path)
import os
save_folder = r"C:\Users\hanst\PycharmProjects\T2Bproject\pdf2array"
for page in markT2B_image:
    img_name = "markT2B.jpeg"
    page.save(os.path.join(save_folder, img_name))
mark = cv.imread(r"C:\Users\hanst\PycharmProjects\T2Bproject\pdf2array\markT2B.jpeg", cv.IMREAD_GRAYSCALE)
print(mark)
from PIL import Image
Image.rotate(mark, 180)

