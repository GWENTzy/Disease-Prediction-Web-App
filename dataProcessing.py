import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from PIL import Image
from function import *

def dataProcessing():    
    st.markdown("<h2 style='text-align: center; color: white;'>Data Processing</h2>", unsafe_allow_html=True)
    st.markdown('<br />', unsafe_allow_html=True)
    st.write('''
    Pada aplikasi ini, kami menggunakan dataset dari (https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) yang terdiri dari 4 dataframe, yaitu:
    1. dataset mengenai penyakit dan gejala - gejalanya
    2. dataset mengenai gejala dan dampaknya ke tubuh setiap dua hari
    3. dataset mengenai penyakit dan cara pencegahannnya
    4. dataset mengenai penyakit dan deskripsi penyakit tersebut
    ''')
    st.markdown('<br />', unsafe_allow_html=True)

    df = pd.read_csv('data/dataset.csv')
    df2 = pd.read_csv('data/symptom-severity.csv')
    df3 = pd.read_csv('data/symptom_precaution.csv')
    df4 = pd.read_csv('data/symptom_Description.csv')

    col1, col2 = st.columns(2)
    with col1:
        st.write('##### Dataframe 1')
        st.write(df)
        st.markdown('<br />', unsafe_allow_html=True)
        st.write('##### Dataframe 3')
        st.write(df3)
    with col2:
        st.write('##### Dataframe 2')
        st.write(df2)
        st.markdown('<br />', unsafe_allow_html=True)
        st.write('##### Dataframe 4')
        st.write(df4)
    st.markdown('<br />', unsafe_allow_html=True)

    st.write('''
    Setelah kami memahami setiap dataframe yang ada, berikutnya kami akan menganalisa dataframe - dataframe tersebut. Namun, karena kami membuat aplikasi untuk memprediksi penyakitnya, maka kami akan menganalisa dataframe 1 dan 2 saja. Sedangkan dataframe 3 dan 4 akan kami gunakan untuk memberikan deskripsi dan penanganan penyakit yang diprediksi
    ''')

    st.subheader('Disease EDA')
    disease_count = df['Disease'].nunique()
    st.success(f'Banyak jenis penyakit pada dataset : {disease_count}')

    fig = plt.figure(figsize=(10, 4))
    chart = sns.barplot(x=df['Disease'].unique(), y=df['Disease'].value_counts(sort = False), data=df, palette='Set3')
    plt.xticks(rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=10)
    chart.set(ylabel='Count')
    chart.set_title('Disease Count Each Category')
    st.pyplot(fig)

    st.write('''
    Dari bar chart di atas dapat dilihat bahwa setiap jenis penyakit terdapat 120 penyakit dengan nama yang sama, tapi dengan gejala yang berbeda - beda. Selain itu, karena semua penyakit memiliki jumlah yang sama, maka distribusi data pada dataset adalah normal
    ''')

    st.subheader('Symptoms EDA')
    symptoms = df.iloc[:, 1:18]
    x = symptoms.columns.values
    y = symptoms.isna().sum()

    fig = plt.figure(figsize=(8,4))
    chart = sns.barplot(x=x, y=y, palette='Set2')
    # chart.axes.get_xaxis().set_visible(False)
    plt.xticks(rotation=45)
    plt.tick_params(axis='x', which='major', labelsize=10)
    chart.set(ylabel='Count')
    chart.set_title('Symptoms NaN Values Count')
    st.pyplot(fig)

    st.write('Dari chart di atas dapat kami simpulkan bahwa gejala yang kosong dimulai dari gejala ketiga dan semakin bertambah nilainya pada gejala selanjutnya. Hal ini disebabkan oleh gejala pada penyakit mayoritas hanya dua sampai tiga gejala utama saja. Dengan demikian jika terdapat gejala penyakit yang lebih dari itu, akan semakin banyak nilai kolom yang kosong')

    col1, col2 = st.columns(2)
    with col1:
        st.write(df2.describe())
    with col2:
        st.write(df2['Symptom'].value_counts(sort=True))
    st.write('''
    Pada tabel kiri terdapat summary value dari kolom weight pada dataframe 2. Dari nilai tersebut dapat disimpulkan bahwa data sangat kurang bervariasi karena nilai standar deviasi jauh lebih kecil dari nilai mean. 
    
    Selain itu, dari tabel kiri terdapat nilai count 133 yang mengindisikan terdapat 133 gejala penyakit yang berbeda pada dataset. Tetapi jika dilihat pada tabel kanan, terdapat satu gejala penyakit yang memiliki nilai yang sama, yaitu "fluid_overload". Jadi total gejala yang sebenarnya adalah 132 gejala penyakit. Nilai yang sama ini nanti akan dihapus salah satunya pada saat melakukan data cleaning
    ''')

    fig = plt.figure(figsize=(24,8))
    chart = sns.barplot(x=df2['Symptom'], y=df2['weight'], data=df2, order=df2.sort_values('weight').Symptom, palette='Set2')
    plt.xticks(rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=10)
    chart.set(ylabel='Weight')
    chart.set_title('Symptoms Weight')
    st.pyplot(fig)
    st.write('''
        Chart di atas menjelaskan bahwa data nilai weight pada dataset berkisar mulai 1 - 7, yang mana kebanyakan berada di nilai weight 4 dan 5. Sedangkan sangat sedikit gejala yang bernilai 1 dan 7. Weight disini adalah nilai untuk mengukur efek gejala terhadap tubuh setiap 2 hari. Semakin tinggi nilainya, maka dampak terhadap tubuh semakin besar pula
    ''')

    fig = plt.figure(figsize=(12,4))
    plt.subplot(1, 3, 1)
    chart = sns.boxplot(x=df2['weight'], data=df2, palette='pastel')
    plt.subplot(1, 3, 2)
    chart = sns.histplot(data=df2, x="weight", kde = True, stat = "probability", discrete = True)
    chart.set_title('Distribusi Kolom Weight pada Dataset 2')
    plt.subplot(1, 3, 3)
    sns.histplot(data=df2, x="weight", kde = True, discrete = True)
    plt.tight_layout()
    st.pyplot(fig)

    st.write('''
        Dari boxplot di chart paling kiri, terlihat jelas bahwa persebaran data sangat normal (normal distribution) dan tidak ada indikasi outlier karena data sangat seimbang antara min, max, Q1, Q2, dan Q3. Pada histogram, seperti yang sudah dijelaskan sebelumnya, nilai yang paling banyak muncul adalah nilai weight 4 dan 5, hal ini menyebabkan probability weight tersebut yang paling tinggi pula
    ''')

    st.subheader('Data Cleaning dan Data Transform')

    st.write('''
    Pada data cleaning dan data transform, terdapat 4 tahap utama yang dijalankan:
    1. Menghapus data ganda --> Hal pertama yang harus dilakukan adalah menghapus data ganda. Untuk itu, kita akan menghapus data "fluid_overload" untuk meningkatkan akurasi model
    2. Mengganti nilai NaN dengan 0 --> Selanjutnya adalah mengganti nilai NaN pada symptom dengan 0. Kenapa bukan mean atau modus? Karena kolom symptom adalah kolom mengenai gejala penyakit, jika nilainya NaN diganti mean atau modus, maka dapat menghasilkan nilai yang salah yang berarti bahwa ada gejala pada kolom tersebut. Oleh sebab itu, NaN mengganti nilai NaN dengan 0
    3. Mengganti string symptom dengan weight dari masing - masing symptom --> machine learning tidak dapat membaca sebuah string, sehingga kita harus mengganti string dengan weight dari masing - masing symptom agar data diubah ke dalam bentuk numerik sesuai dengan data yang ada pada dataset 2
    4. Mengganti string symptom yang memiliki value berbeda ke value sebenarnya, kemudian merubahnya ke weight symptom tersebut --> Ternyata terdapat beberapa value dari kolom yang typo, yaitu kelebihan spasi (ex: 'dischromic _patches', 'foul_smell_of urine', dan 'spotting_ urination'). Oleh sebab itu, kami menghapus spasi yang ada dan mengubahnya ke dalam bentuk numerik melalui weight yang ada pada dataset 2
    ''')
    st.write('Raw Data:')
    st.write(df)
    final_df = clean_data(df, df2)
    st.write('Clean Data:')
    st.write(final_df)

    st.subheader('Extract Dependent and Independent Variables')
    st.write('Berikutnya adalah membuat variabel independen atau X, dan dependen atau y. Variabel independennya adalah semua gejala pada penyakit tertentu. Sedangkan variabel dependennya adalah penyakit itu sendiri karena produk kami adalah aplikasi prediksi penyakit. Variabel dependen sangat bergantung pada nilai variabel independen, sama seperti penyakit yang bisa didiagnosis setelah mengetahui gejala - gejalanya')
    X = final_df.iloc[:, 1:]
    y = final_df['Disease']
    X_shape = X.shape
    y_shape = y.shape
    st.success(f'X shape : {X_shape}')
    st.success(f'y shape : {y_shape}')
    st.write('Dataframe X:')
    st.write(X)
    st.write('Dataframe y:')
    st.write(y)

    st.subheader('Dataset Splitting')
    st.write('Untuk membangun model, dibutuhkan training dan testing data. Training digunakan untuk melatih model dengan data yang sudah ada, sedangkan testing digunakan untuk melihat hasil model dengan data yang belum pernah dilihat atau dilatih sebelumnya. Data akan dibagi dengan skala 4 : 1, yaitu 80 persen untuk training set dan 20 persen untuk testing set. User juga bisa mengubah nilai test_size dan random_state di bagian HOME')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f'X_train {X_train.shape}')
        st.write(X_train)
    with col2:
        st.write(f'X_test: {X_test.shape}')
        st.write(X_test)
    with col3:
        st.write(f'y_train: {y_train.shape}')
        st.write(y_train)
    with col4:
        st.write(f'y_test: {y_test.shape}')
        st.write(y_test)
    
    st.subheader('Model Selection')
    st.write('Algoritma model yang bisa dipilih user adalah KNN, yaitu model untuk mencari nilai terdekat dari nilai prediksi untuk menentukan hasil prediksinya, dan algoritma Random Forest Classification, yaitu algoritma klasifikasi untuk mencari nilai voting tertinggi dari setiap tree yang ada pada forest atau kumpulan tree untuk menentukan hasil prediksinya. Berdasarkan hasil akurasi dari model, bisa dilihat bahwa random forest memberikan hasil yang sedikit lebih baik dengan default parameter untuk kedua algoritma ini. User juga bisa memanipulasi nilai dari parameter tiap algoritma di bagiam HOME')
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    model2 = RandomForestClassifier()
    model2.fit(X_train, y_train)

    y_pred_1 = model.predict(X_test)
    accuracy_score_1 = accuracy_score(y_test, y_pred_1)*100

    y_pred_2 = model2.predict(X_test)
    accuracy_score_2 = accuracy_score(y_test, y_pred_2)*100

    st.write(f'KNN Accuracy: {accuracy_score_1}')
    st.write(f'Random Forest Accuracy: {accuracy_score_2}')
    evaluation_df = pd.DataFrame()
    evaluation_df['Model'] = ['KNN', 'Random Forest']
    evaluation_df['Accuracy'] = [accuracy_score_1, accuracy_score_2]

    fig = plt.figure(figsize=(12, 5))
    chart = sns.barplot(data=evaluation_df, x=evaluation_df['Model'], y=evaluation_df['Accuracy'], palette="Set2")
    chart.set_title('Evaluasi Akurasi Model')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader('Features Importance')
    st.write('Karena fitur independen yang memiliki nilai paling besar adalah Symptom_1 sampai Symptom_6, maka fitur saya potong dari yang awalnya 17 menjadi maksimal 6. Hal ini juga masih masuk akal, mengingat gejala yang biasa dikeluhkan user hanya beberapa saja, jarang ada user memberi gejala lebih dari 6 gejala')

    importances = model2.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model2.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=X_train.columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    st.pyplot(fig)
