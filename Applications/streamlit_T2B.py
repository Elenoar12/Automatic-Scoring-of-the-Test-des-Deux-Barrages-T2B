from T2B_functions import *
import streamlit as st

# Start GUI to select file and specify last processed symbol
st.title('T2B leicht gemacht')

st.write('**Test des Deux Barrages auswählen:**')
file_path = st.file_uploader('Durchsuchen', type=["pdf"], label_visibility='collapsed')

st.write('**Zeile und Spalte des letzten bearbeiteten Symbols:**')
row = st.text_input('Zeile')
col = st.text_input('Spalte')

if st.button('Bestätigen'):
    if file_path is None:
        st.error('Kein Testblatt ausgewählt')
    # Uncomment if filepath is necessary and may cause an error
    #elif not os.path.exists(file_path.name):
        #st.error('Die ausgewählte Datei existiert nicht')
    elif not row or not col:
        st.error('Zeilen- oder Spaltenwert fehlt')
    else:
        try:
            row = int(row)
            col = int(col)
            if 1 <= row <= 40 and 1 <= col <= 25:
                st.success('Auswertung beginnt')
                stop = [row, col]
                st.write(f'Testblatt: {file_path.name}')
                st.write(f'Letztes bearbeitetes Symbol: {stop}')
                file = file_path.getvalue()
                progress = st.progress(0, text='Vorbereitung läuft..')
                t2b = pre_process(file)
                progress.progress(33, text='Wird verarbeitet..')
                img, mark, template, adapt_row, adapt_clmn = process_t2b(t2b, stop)
                progress.progress(66, text='Wird ausgewertet..')
                str_mark = visual(img, mark, template, adapt_row, adapt_clmn)[1]
                temp_excel_path = excel_gen(str_mark)
                with open(temp_excel_path, 'rb') as excel_file:
                    excel_bytes = excel_file.read()
                progress.progress(100, text='Fertig!')
                # Add a button for the user to download the file
                st.write('**Auswertung herunterladen**')
                st.download_button(
                            label='Herunterladen',
                            data=excel_bytes,
                            key='excel_file',
                            file_name='T2B leicht gemacht.xlsm',
                            mime='application/octet-stream')
            else:
                st.error('Zeilen- oder Spaltenwert nicht im gültigen Bereich')
        except ValueError:
            st.error('Zeilen- oder Spaltenwert ist keine Zahl')