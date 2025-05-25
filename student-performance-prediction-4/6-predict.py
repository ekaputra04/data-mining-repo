import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import joblib
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Definisi Path (Ganti dengan absolute path di sistem Anda)

BASE_URL = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_URL, 'xgboost_model.pkl')
XTRAIN_PATH = os.path.join(BASE_URL, 'dataset', 'X_train.csv')

# 2. Muat Model dan X_train
xgb_model = joblib.load(MODEL_PATH)
X_train = pd.read_csv(XTRAIN_PATH)

# 3. Definisi Kolom
raw_columns = [
    'age', 'gender', 'major', 'study_hours_per_day', 'social_media_hours', 
    'netflix_hours', 'part_time_job', 'attendance_percentage', 'sleep_hours', 
    'diet_quality', 'exercise_frequency', 'parental_education_level', 
    'internet_quality', 'mental_health_rating', 'extracurricular_participation', 
    'previous_gpa', 'semester', 'stress_level', 'dropout_risk', 'social_activity', 
    'screen_time', 'study_environment', 'access_to_tutoring', 'family_income_range', 
    'parental_support_level', 'motivation_level', 'exam_anxiety_score', 
    'learning_style', 'time_management_score'
][:-1]  # Hapus 'exam_score'

numerical_cols = [
    'age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 
    'attendance_percentage', 'sleep_hours', 'exercise_frequency', 
    'mental_health_rating', 'previous_gpa', 'semester', 'stress_level', 
    'social_activity', 'screen_time', 'parental_support_level', 
    'motivation_level', 'exam_anxiety_score', 'time_management_score'
]
boolean_cols = ['part_time_job', 'extracurricular_participation', 'access_to_tutoring']
nominal_cols = ['gender', 'major', 'study_environment']
ordinal_cols = [
    'diet_quality', 'parental_education_level', 'internet_quality', 
    'dropout_risk', 'family_income_range', 'learning_style'
]

# Kategori berdasarkan kolom X_train dan definisi preprocessing
categories = {
    'gender': ['Female', 'Male', 'Other'],
    'major': ['Mathematics', 'Biology', 'Business', 'Computer Science', 'Engineering', 'Psychology'],
    'study_environment': ['Home', 'Co-Learning Group', 'Dorm', 'Library', 'Quiet Room'],
    'diet_quality': ['Poor', 'Average', 'Good'],
    'parental_education_level': ['High School', 'Bachelor', 'Master', 'PhD'],
    'internet_quality': ['Poor', 'Average', 'Good'],
    'dropout_risk': ['Low', 'Medium', 'High'],
    'family_income_range': ['Low', 'Medium', 'High'],
    'learning_style': ['Auditory', 'Kinesthetic', 'Reading', 'Visual']
}

# 4. Buat Objek Preprocessing
# StandardScaler untuk numerik
scaler = StandardScaler()
scaler.fit(X_train[numerical_cols])

# OneHotEncoder untuk nominal
onehot_encoder = OneHotEncoder(
    categories=[categories[col] for col in nominal_cols],
    sparse_output=False,
    drop='first',
    handle_unknown='ignore'
)
dummy_data = pd.DataFrame({col: [categories[col][0]] for col in nominal_cols})
onehot_encoder.fit(dummy_data)

# LabelEncoder untuk ordinal
label_encoders = {}
for col in ordinal_cols:
    le = LabelEncoder()
    le.fit(categories[col])
    label_encoders[col] = le

# 5. Hitung Mean, Modus, dan IQR dari X_train
means = X_train[numerical_cols].mean()
modes = {
    'part_time_job': 0,
    'extracurricular_participation': 0,
    'access_to_tutoring': 0,
    'gender': 'Female',
    'major': 'Mathematics',
    'study_environment': 'Home',
    'diet_quality': 'Average',
    'parental_education_level': 'Bachelor',
    'internet_quality': 'Average',
    'dropout_risk': 'Low',
    'family_income_range': 'Medium',
    'learning_style': 'Visual'
}

# IQR untuk outlier
iqr_bounds = {}
for col in numerical_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    iqr_bounds[col] = {'lower': Q1 - 1.5 * IQR, 'upper': Q3 + 1.5 * IQR}

# 6. Fungsi untuk Preprocessing Satu Data
def preprocess_single_data(input_data, raw_columns, means, modes, iqr_bounds):
    data = pd.DataFrame([input_data], columns=raw_columns)
    
    # Tangani nilai hilang
    for col in numerical_cols:
        if col in data.columns and pd.isna(data[col].iloc[0]):
            data[col] = means[col]
    
    for col in boolean_cols + nominal_cols + ordinal_cols:
        if col in data.columns and pd.isna(data[col].iloc[0]):
            data[col] = modes[col]
    
    # Tangani outlier untuk kolom numerik
    for col in numerical_cols:
        if col in data.columns:
            value = data[col].iloc[0]
            lower = iqr_bounds.get(col, {'lower': -np.inf})['lower']
            upper = iqr_bounds.get(col, {'upper': np.inf})['upper']
            data[col] = np.clip(value, lower, upper)
    
    # Boolean Encoding
    for col in boolean_cols:
        if col in data.columns:
            value = str(data[col].iloc[0]).strip()  # Normalisasi ke string, hapus spasi
            logging.info(f"Processing boolean column {col} with value '{value}' (type: {type(value)})")
            if value in ['Yes', 'No', 'True', 'False', '1', '0']:
                data[col] = 1 if value in ['Yes', 'True', '1'] else 0
            else:
                print(f"Warning: Nilai '{value}' di kolom {col} tidak dikenali. Menggunakan modus.")
                data[col] = modes[col]
    
    # One-Hot Encoding untuk Nominal
    if nominal_cols:
        nominal_data = data[nominal_cols]
        encoded_nominal = onehot_encoder.transform(nominal_data)
        encoded_nominal_df = pd.DataFrame(
            encoded_nominal,
            columns=onehot_encoder.get_feature_names_out(nominal_cols)
        )
        data = data.drop(columns=nominal_cols)
        data = pd.concat([data, encoded_nominal_df], axis=1)
    
    # Label Encoding untuk Ordinal
    for col in ordinal_cols:
        if col in data.columns:
            try:
                data[col] = label_encoders[col].transform([data[col].iloc[0]])[0]
            except ValueError:
                print(f"Warning: Nilai '{data[col].iloc[0]}' di kolom {col} tidak dikenali. Menggunakan modus.")
                data[col] = label_encoders[col].transform([modes[col]])[0]
    
    # Normalisasi fitur numerik
    if numerical_cols:
        data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    # Pastikan urutan kolom sesuai X_train
    data = data[X_train.columns]
    
    return data

# 7. Input Pengguna
print("Masukkan data untuk prediksi exam_score:")
input_data = {}
for col in raw_columns:
    while True:
        try:
            prompt = f"Masukkan {col} (tekan Enter untuk menggunakan nilai default"
            if col in boolean_cols:
                prompt += ", masukkan 'Yes', 'No', 'True', 'False', '1', atau '0')"
            elif col in nominal_cols + ordinal_cols:
                prompt += f", pilih dari {categories[col]}"
            else:
                prompt += ")"
            value = input(f"{prompt}: ")
            logging.info(f"Raw input untuk {col}: '{value}' (type: {type(value)})")
            if value.strip() == "":
                input_data[col] = np.nan
            else:
                if col in numerical_cols:
                    input_data[col] = float(value)
                elif col in boolean_cols:
                    if value not in ['Yes', 'No', 'True', 'False', '1', '0']:
                        raise ValueError(f"Nilai '{value}' tidak valid untuk {col}. Gunakan 'Yes', 'No', 'True', 'False', '1', atau '0'.")
                    input_data[col] = value
                elif col in nominal_cols + ordinal_cols:
                    if value not in categories[col]:
                        raise ValueError(f"Nilai '{value}' tidak valid untuk {col}. Pilih dari {categories[col]}.")
                    input_data[col] = value
                else:
                    input_data[col] = value
            break
        except ValueError as e:
            print(f"Input tidak valid: {e}. Coba lagi.")

# 8. Preprocessing Input
processed_data = preprocess_single_data(input_data, raw_columns, means, modes, iqr_bounds)

# 9. Prediksi
prediction = xgb_model.predict(processed_data)[0]

# 10. Tampilkan Hasil
print("\n=== Hasil Prediksi ===")
print(f"Prediksi Exam Score: {prediction:.2f}")