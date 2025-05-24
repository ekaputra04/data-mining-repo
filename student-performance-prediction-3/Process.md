# Gambaran Proses Penelitian

## 1. Pemahaman Dataset dan Tujuan

`Tujuan`: Memprediksi Total_Score (variabel kontinu, weighted sum of all grades) mahasiswa menggunakan lima metode regresi dan membandingkan performanya.

`Dataset`: Berdasarkan metadata, dataset memiliki 21 kolom, termasuk fitur akademik (Attendance, Midterm_Score, dll.), perilaku (Study_Hours_per_Week, Sleep_Hours_per_Night), dan latar belakang (Gender, Family_Income_Level, dll.).

`Output`: Perbandingan performa model berdasarkan metrik evaluasi (MAE, RMSE, R²) dan identifikasi faktor utama yang memengaruhi Total_Score.

## 2. Preprocessing Data

Preprocessing data sangat penting untuk memastikan dataset siap digunakan oleh model regresi. Langkah-langkahnya adalah sebagai berikut:

### 2.1. Pemeriksaan Missing Values:

Periksa apakah ada nilai yang hilang pada kolom seperti Total_Score, Attendance, atau Study_Hours_per_Week.

Penanganan:

- Untuk fitur numerik (misalnya Midterm_Score, Assignments_Avg): Imputasi dengan mean atau median.
- Untuk fitur kategorikal (misalnya Gender, Department): Imputasi dengan mode atau nilai paling sering.
- Jika missing values sangat sedikit, pertimbangkan untuk menghapus baris tersebut.

### 2.2. Encoding Variabel Kategorikal:

Kolom kategorikal: Gender, Department, Parent_Education_Level, Family_Income_Level, Grade, Extracurricular_Activities, Internet_Access_at_Home.
Metode encoding:

- One-Hot Encoding: Untuk kolom seperti Gender (Male, Female, Other), Department, Parent_Education_Level, dan Family_Income_Level, karena jumlah kategori terbatas.
- Label Encoding: Untuk kolom ordinal seperti Parent_Education_Level (None, High School, Bachelor's, Master's, PhD) jika urutan dianggap relevan.
- Boolean Encoding: Untuk Extracurricular_Activities dan Internet_Access_at_Home (Yes/No menjadi 1/0).
- Catatan: Kolom Grade tidak digunakan sebagai fitur karena merupakan hasil turunan dari Total_Score.

### 2.3. Normalisasi/Standarisasi Fitur Numerik:

- Fitur numerik: Attendance (%), Midterm_Score, Final_Score, Assignments_Avg, Quizzes_Avg, Participation_Score, Projects_Score, Study_Hours_per_Week, Sleep_Hours_per_Night, Age, Stress_Level.
- Terapkan StandardScaler (standarisasi ke mean=0, std=1) untuk model seperti SVR dan Ridge Regression yang sensitif terhadap skala data.
- MinMaxScaler (skala ke rentang 0-1) dapat digunakan sebagai alternatif untuk memastikan semua fitur berada dalam rentang yang seragam.

### 2.4. Pemeriksaan Outlier:

- Identifikasi outlier pada fitur numerik (misalnya Study_Hours_per_Week yang sangat tinggi atau Total_Score yang tidak wajar) menggunakan metode seperti IQR atau Z-score.
- Penanganan: Pertimbangkan untuk membatasi outlier (capping) atau menghapusnya jika tidak realistis.

### 2.5. Pemilihan Fitur Awal:

- Fitur yang akan digunakan:

  - Akademik: Attendance, Midterm_Score, Final_Score, Assignments_Avg, Quizzes_Avg, Participation_Score, Projects_Score.
  - Perilaku: Study_Hours_per_Week, Sleep_Hours_per_Night, Extracurricular_Activities.
  - Latar Belakang: Gender, Age, Department, Parent_Education_Level, Family_Income_Level, Internet_Access_at_Home.
  - Psikologis: Stress_Level.

- Pertimbangkan korelasi antar fitur (misalnya Midterm_Score dan Final_Score mungkin berkorelasi tinggi) untuk menghindari multikolinearitas, terutama pada Linear Regression dan Ridge Regression.
- Gunakan teknik seperti Pearson Correlation atau Variance Inflation Factor (VIF) untuk mendeteksi multikolinearitas.

### 2.6. Pembagian Data:

- Pisahkan dataset menjadi data pelatihan (80%) dan data pengujian (20%) menggunakan train-test split.
- Gunakan stratified sampling (jika memungkinkan) untuk memastikan distribusi fitur kategorikal seperti Department seimbang.
- Terapkan 5-fold cross-validation pada data pelatihan untuk mengevaluasi model secara robust.
