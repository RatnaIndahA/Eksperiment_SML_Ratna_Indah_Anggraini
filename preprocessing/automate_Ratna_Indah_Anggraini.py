# preprocessing/automate_Ratna_Indah_Anggraini.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

def preprocess_adult_data(input_path: str, output_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Melakukan preprocessing lengkap pada dataset Adult Census Income.
    Hasil akhir: CSV dengan fitur hasil one-hot encoding + target 'income'.
    """
    # 1. Load data
    df = pd.read_csv(input_path)

    # 2. Hapus duplikat
    df.drop_duplicates(inplace=True)

    # 3. Konversi '?' menjadi NaN
    df.replace('?', np.nan, inplace=True)

    # 4. Pisahkan fitur dan target
    X = df.drop(columns=['income'])
    y = df['income']

    # 5. Identifikasi kolom
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 6. Pipeline numerik
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 7. Pipeline kategorikal
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 8. ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # 9. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 10. Transformasi
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    # 11. Nama kolom hasil encoding
    encoded_cat_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    all_features = numerical_cols + list(encoded_cat_cols)

    # 12. DataFrame hasil
    X_train_df = pd.DataFrame(X_train_ready, columns=all_features)
    X_test_df = pd.DataFrame(X_test_ready, columns=all_features)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_final = pd.concat([X_train_df, y_train], axis=1)
    test_final = pd.concat([X_test_df, y_test], axis=1)

    df_clean_total = pd.concat([train_final, test_final], ignore_index=True)

    # 13. Simpan
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean_total.to_csv(output_path, index=False)
    print(f"Preprocessing selesai. Data bersih disimpan di: {output_path}")
    print(f"Shape: {df_clean_total.shape}")
    return df_clean_total

if __name__ == "__main__":
    # Sesuai dengan struktur repository
    input_file = "../adult_raw.csv"
    output_file = "preprocessing/adult_preprocessing.csv"
    preprocess_adult_data(input_file, output_file)