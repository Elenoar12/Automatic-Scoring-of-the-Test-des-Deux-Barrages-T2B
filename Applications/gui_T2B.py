import PySimpleGUI as sg
import os
from T2B_functions import *

#function to preprocess T2B scans (conversion, rotation and skew correction)
def pre_process(file_path):
    # Convert pdf to grayscale image
    t2b = convert_from_path(file_path)
    t2b = cv2.cvtColor(np.array(t2b[0]), cv2.COLOR_BGR2GRAY)
    dim = t2b.shape
    # Read the template
    template = cv2.imread(r"C:\Users\hanst\PycharmProjects\scan_T2B\target_symbols.png", 0)
    # Perform match operations.
    res = cv2.matchTemplate(t2b, template, cv2.TM_CCOEFF_NORMED)
    # Specify a threshold
    threshold = 0.1
    # Store the coordinates of the matched area in a numpy array
    loc = np.where(res >= threshold)
    # Rotate test sheet depending on where the matched area was found
    if np.mean(loc[1]) < (dim[1] / 2):
        t2b = cv2.rotate(t2b, cv2.ROTATE_90_CLOCKWISE)
    else:
        t2b = cv2.rotate(t2b, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Prepare the test sheet for skew correction (for imgreg: arrays must have the same number of rows and columns)
    ref_t2b = cv2.imread(r"C:\Users\hanst\PycharmProjects\scan_T2B\T2B.jpeg", cv2.IMREAD_GRAYSCALE)
    # Get the number of rows and columns in each array
    num_rows_t2b, num_cols_t2b = t2b.shape
    num_rows_ref, num_cols_ref = ref_t2b.shape
    # Determine the difference in the number of rows and columns
    row_difference = num_rows_ref - num_rows_t2b
    col_difference = num_cols_ref - num_cols_t2b
    # Adjust the rows and columns of arr1
    if row_difference > 0:
        t2b = np.pad(t2b, ((0, row_difference), (0, 0)), mode='constant', constant_values=255)
    elif row_difference < 0:
        ref_t2b = np.pad(ref_t2b, ((0, -row_difference), (0, 0)), mode='constant', constant_values=255)
    if col_difference > 0:
        t2b = np.pad(t2b, ((0, 0), (0, col_difference)), mode='constant', constant_values=255)
    elif col_difference < 0:
        ref_t2b = np.pad(ref_t2b, ((0, 0), (0, -col_difference)), mode='constant', constant_values=255)
    # Determine whether the test sheet needs rotation
    rtt = LogPolarSolver(ref_t2b, t2b).RECOVERED_ROTATION.value
    t2b = rotate(t2b, -rtt[0], reshape=False)
    return t2b

#function to visualize missed target symbols and marked non-target symbols:
def visual(img, mark, template, adapt_row, adapt_clmn):
  FN_loc = np.logical_and(mark == 0, template == 1)
  FN_indx = np.transpose(np.nonzero(FN_loc == True))
  FP_loc = np.logical_and(mark == 1, template == 0)
  FP_indx = np.transpose(np.nonzero(FP_loc == True))
  #If excel paste output preferred:
  str_mark = mark.astype(str)
  str_mark[FN_indx[:, 0], FN_indx[:, 1]] = 'a'
  str_mark[FP_indx[:, 0], FP_indx[:, 1]] = 'f'
  str_mark[str_mark == '0'] = ''
  str_mark[str_mark == '1'] = ''
  #Remove the first row to paste onto excel worksheet:
  str_mark = str_mark[1:]
  #visualization:
  vis = plt.figure(figsize=(12, 7))
  plt.axis('off')
  plt.imshow(img, cmap='gray')
  for w in range(len(FN_indx)):
    plt.hlines((adapt_row[FN_indx[w, 0]], adapt_row[(FN_indx[w, 0] + 1)]), adapt_clmn[FN_indx[w, 1]], adapt_clmn[(FN_indx[w, 1] + 1)], colors = ("orange"), linewidth = 1)
    plt.vlines((adapt_clmn[FN_indx[w, 1]], adapt_clmn[(FN_indx[w, 1] + 1)]), adapt_row[FN_indx[w, 0]], adapt_row[(FN_indx[w, 0] + 1)], colors = ("orange"), linewidth = 1)
  for v in range(len(FP_indx)):
    plt.hlines((adapt_row[FP_indx[v, 0]], adapt_row[(FP_indx[v, 0] + 1)]), adapt_clmn[FP_indx[v, 1]], adapt_clmn[(FP_indx[v, 1] + 1)], colors = ("r"), linewidth = 1)
    plt.vlines((adapt_clmn[FP_indx[v, 1]], adapt_clmn[(FP_indx[v, 1] + 1)]), adapt_row[FP_indx[v, 0]], adapt_row[(FP_indx[v, 0] + 1)], colors = ("r"), linewidth = 1)
  plt.show()
  return vis, str_mark

def excel_gen(str_mark):
    # Open the Excel file
    excel_path = r"C:\Users\hanst\PycharmProjects\scan_T2B\Profil MNND.xlsm"
    wb = xw.Book(excel_path)
    # Select the 'T2B' worksheet
    ws = wb.sheets['T2B']
    # Replace the values in the specified range (AA5 to AY43)
    ws.range('AA5').options(expand='table').value = str_mark
    # Save the modified workbook
    eval_path = r'C:\Users\hanst\PycharmProjects\scan_T2B\T2B leicht gemacht.xlsm'
    wb.save(eval_path)
    wb.app.quit()

#function to start GUI application for file input to export
def GUI_start():
    sg.theme('DefaultNoMoreNagging')

    layout = [
        [sg.Text('Test des Deux Barrages auswählen:')],
        [sg.FileBrowse('Durchsuchen', file_types=(("PDF Files", "*.pdf"),)), sg.Input('', key='-FILE-', enable_events=True)],
        [sg.Text('Zeile und Spalte des letzten bearbeiteten Symbol:')],
        [sg.InputText('', key='Zeile', size=(10, 1)), sg.InputText('', key='Spalte', size=(10, 1))],
        [sg.Button('Bestätigen'), sg.Button('Abbrechen')],
    ]

    window = sg.Window('T2B leicht gemacht', layout, finalize=True)
    file_path = None
    stop = None
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Abbrechen':
            break
        elif event == '-FILE-':
            file_path = values['-FILE-']
            if not file_path:
                sg.popup('Kein Testblatt ausgewählt', button_color=('white', 'red'), custom_text='Wiederholen', no_titlebar=True)
            elif not os.path.isfile(file_path):
                sg.popup('Die ausgewählte Datei existiert nicht', button_color = ('white', 'red'), custom_text='Wiederholen', no_titlebar=True)
        elif event == 'Bestätigen':
            x = values['Zeile']
            y = values['Spalte']
            if not file_path:
                sg.popup('Kein Testblatt ausgewählt', button_color=('white', 'red'), custom_text='Wiederholen', no_titlebar=True)
            elif not x or not y:
                sg.popup('Zeilen oder Spaltenwert fehlt', button_color=('white', 'red'), custom_text='Wiederholen', no_titlebar=True)
            else:
                try:
                    x = int(x)
                    y = int(y)
                    if 1 <= x <= 40 and 1 <= y <= 25:
                        stop = [x, y]
                        t2b = pre_process(file_path)
                        img, mark, template, adapt_row, adapt_clmn = process_t2b(t2b, stop)
                        str_mark = visual(img, mark, template, adapt_row, adapt_clmn)[1]
                        window.close()
                        break
                    else:
                        sg.popup('Zeilen- oder Spaltenwert nicht im gültigen Bereich', button_color=('white', 'red'), custom_text='Wiederholen', no_titlebar=True)
                except ValueError:
                    sg.popup('Zeilen- oder Spaltenwert ist keine Zahl', button_color=('white', 'red'), custom_text='Wiederholen', no_titlebar=True)
    return str_mark

if __name__ == "__main__":
    str_mark = GUI_start()
    excel_gen(str_mark)