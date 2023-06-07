import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw() #use to hide tkinter window
def search_for_file_path():
    filepath = filedialog.askopenfilename(parent = root, title = 'Select test sheet to convert', filetypes = (("PDF files","*.pdf"),("all files","*.*")))
    return filepath
filepath = search_for_file_path()
pdf_path = filepath

from pdf2image import convert_from_path #installing poppler package necessary
T2B_img = convert_from_path(pdf_path)
import os
save_folder = r"C:\Users\hanst\PycharmProjects\T2B\pdf2array"                                                           #decide where to save image
for page in T2B_img:
    pdf_name = os.path.basename(filepath)
    img_name = os.path.splitext(pdf_name)[0] + ".jpeg"
    img_path = page.save(os.path.join(save_folder, img_name))
