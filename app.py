import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from kmeans import kmeans_and_logistic_regression
from kmedian import k_median_clustering
import numpy as np
import time
from logres import logres
from naivebayes import naivbay


    

with st.sidebar:
    st.image("cloud.png")
    st.title("Uji CLustering Data")
    pilihan = st.radio("", ["Home", "Upload File", "Uji Data"])
    st.info("Pastikan data anda sudah siap untuk dilakukan pengujian dengan metode clustering")

if os.path.exists("data.csv"):
    source = pd.read_csv("data.csv")

if pilihan == "Home":
    if 'nama' not in st.session_state:
        st.session_state.nama = ''
    def submit_name():
        st.session_state.nama = st.session_state.temp_nama

    if st.session_state.nama == '':
        st.text_input("Masukkan nama anda:", key='temp_nama', on_change=submit_name)
    else:
        st.title(f"Selamat datang {st.session_state.nama} di Aplikasi Pengelompokan Data!")

        st.markdown("""
        Aplikasi ini memungkinkan Anda untuk melakukan beberapa algoritma machine learning tingkat dasar.
        
        **Petunjuk:**
        
        1. **Unggah File:** Klik opsi "Unggah File" di sisi untuk mengunggah file data CSV Anda.
        
        2. **Praproses Data:** Setelah mengunggah file, Anda dapat memilih kolom yang ingin dihapus jika diperlukan.
        
        3. **Lakukan Pengelompokan:** Setelah data Anda siap, navigasikan ke bagian "Uji Data" untuk menentukan parameter dan melakukan pengelompokan.
        
        4. **Lihat Hasil:** Hasilnya, termasuk visualisasi pengelompokan dan akurasi, akan ditampilkan.
        
        Nikmati eksplorasi data Anda dengan pengelompokan!
        """)

        if 'source' in globals() and not source.empty:
            st.subheader("Contoh Visualisasi")
            with st.spinner("Memuat Konten..."):
                time.sleep(10)
            fig, ax = plt.subplots()

            colors = np.random.rand(len(source))

            sizes = 100 * np.random.rand(len(source))

            scatter = ax.scatter(
                source.iloc[:, 0], source.iloc[:, 1], 
                c=colors, cmap='viridis', s=sizes, alpha=0.7, edgecolor='w', linewidth=0.5
            )

            colorbar = plt.colorbar(scatter, ax=ax)
            colorbar.set_label('Random Colors', color='white')

            ax.set_xlabel(source.columns[0], color='white')
            ax.set_ylabel(source.columns[1], color='white')
            ax.set_title('Scatter Plot', color='white')

            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

            colorbar.ax.yaxis.set_tick_params(color='white')

            colorbar.ax.yaxis.set_tick_params(labelcolor='white')

            fig.patch.set_facecolor((0, 0, 0, 0.0)) 
            ax.set_facecolor((0.2, 0.4, 0.6, 0.1)) 

            st.pyplot(fig)
        else:
            st.write("Unggah dataset untuk melihat visualisasi dan akurasi.")

if pilihan == "Upload File":
    file = st.file_uploader("Upload your file")
    if file is not None:
        ex = pd.read_csv(file, index_col=None)
        st.write(ex)
        dropKolom = st.multiselect("Pilih kolom yang mau di hapus:", ex.columns)
        simpan = st.button("Simpan")
        if dropKolom:
            setelahDrop = ex.drop(dropKolom, axis=1)
            st.dataframe(setelahDrop)
            if simpan:
                namaFile = 'data.csv'
                setelahDrop.to_csv(namaFile, index=None)
                st.success(f"Berhasil menyimpan file {namaFile} ")
        else:
            if simpan:
                namaFile = 'data.csv'
                ex.to_csv(namaFile, index=None)
                st.success(f"Berhasil menyimpan file {namaFile} ")
if pilihan == "Uji Data":
    algo = st.selectbox("Pilih Metode", ["K-Means", "K-Median", "Logistic Regression", "Naive Bayes"])
    if algo == "K-Means":
        st.title("K-Means")
        nilaiK=st.number_input("Masukkan Nilai K: ", min_value=2, value=3)
        if 'source' in globals():
            columns = source.columns.tolist()
            fitur1_index = st.selectbox("Pilihlah label X:", columns)
            fitur2_index = st.selectbox("Pilihlah label Y:", columns, index=1)
            indexfiture1  = 0
            indexfiture2 = 0
            for i in range(len(columns)):
                if columns[i] == fitur1_index:
                    indexfiture1 = i
                elif columns[i] == fitur2_index:
                    indexfiture2 = i
        else:
            st.write("Data tidak ditemukan, silahkan upload data terlebih dahulu")

        if st.button("Ujikan"):
                if nilaiK <= 1:
                    st.error("Nilai K harus lebih dari 1")
                else:
                    if 'source' in globals():
                        with st.spinner("Memuat Konten..."):
                            time.sleep(2)
                        kmeans_and_logistic_regression(source, nilaiK, indexfiture1, indexfiture2)
                    else:
                        st.write("Data tidak ditemukan, silahkan upload data terlebih dahulu")
    elif algo == "K-Median":
        st.title("K-Median")
        nilaiK=st.number_input("Masukkan Nilai K: ", min_value=2, value=3)
        if 'source' in globals():
            columns = source.columns.tolist()
            fitur1_index = st.selectbox("Pilihlah label X:", columns)
            fitur2_index = st.selectbox("Pilihlah label Y:", columns, index=1)
            indexfiture1  = 0
            indexfiture2 = 0
            for i in range(len(columns)):
                if columns[i] == fitur1_index:
                    indexfiture1 = i
                elif columns[i] == fitur2_index:
                    indexfiture2 = i
        else:
            st.write("Data tidak ditemukan, silahkan upload data terlebih dahulu")

        if st.button("Ujikan"):
                if nilaiK <= 1:
                    st.error("Nilai K harus lebih dari 1")
                else:
                    if 'source' in globals():
                        with st.spinner("Memuat Konten..."):
                            time.sleep(2)
                        k_median_clustering(source, nilaiK, indexfiture1, indexfiture2)
                    else:
                        st.write("Data tidak ditemukan, silahkan upload data terlebih dahulu")
    elif algo == "Logistic Regression":
        st.title("Logistic Regression")
        kolom = source.columns.tolist()
        labeling = st.selectbox("Pilihlah Target:", kolom)
        test_size_options = {
        "10%": 0.1, "20%": 0.2, "30%": 0.3, "40%": 0.4, "50%": 0.5,
        "60%": 0.6, "70%": 0.7, "80%": 0.8, "90%": 0.9, "100%": 1.0
        }        
        testUkuran = st.selectbox("Ukuran Uji Data", list(test_size_options.keys()))
        st.write(source)
        if st.button("Ujikan"):
            if 'source' in globals():
                testUkuran = test_size_options[testUkuran]
                logres(source,labeling, testUkuran)
            else:
                st.write("Data tidak ditemukan, silahkan upload data terlebih dahulu")
        st.title("Naive Bayes")
        kolom = source.columns.tolist()
        labeling = st.selectbox("Pilihlah Target:", kolom)
        test_size_options = {
        "10%": 0.1, "20%": 0.2, "30%": 0.3, "40%": 0.4, "50%": 0.5,
        "60%": 0.6, "70%": 0.7, "80%": 0.8, "90%": 0.9, "100%": 1.0
        }        
        testUkuran = st.selectbox("Ukuran Uji Data", list(test_size_options.keys()))
        st.write(source)
        if st.button("Ujikan"):
            if 'source' in globals():
                testUkuran = test_size_options[testUkuran]
                naivbay(source,labeling, testUkuran)
            else:
                st.write("Data tidak ditemukan, silahkan upload data terlebih dahulu")

    
    




