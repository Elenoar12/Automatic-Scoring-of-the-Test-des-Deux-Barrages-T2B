import tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw() #use to hide tkinter window
def search_for_file_path():
    filepath = filedialog.askopenfilename(parent = root, title = 'Please select a file', filetypes = (("PDF files","*.pdf"),("all files","*.*")))
    return filepath
filepath = search_for_file_path()
pdf_path = filepath

#Image.MAX_IMAGE_PIXELS = 1000000000 #plötzlich angeben müssen wtf
from pdf2image import convert_from_path #installing poppler package necessary
T2B_img = convert_from_path(pdf_path)
import os
save_folder = r"C:\Users\hanst\PycharmProjects\T2B\pdf2array"
for page in T2B_img:
    pdf_name = os.path.basename(filepath)
    img_name = os.path.splitext(pdf_name)[0] + ".jpeg"
    img_path = page.save(os.path.join(save_folder, img_name))

jpeg_path = os.chdir(save_folder)
from PIL import ImageTk, Image
from tkinter import Frame, Tk, Label
