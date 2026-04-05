import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Tu dong tai tai nguyen
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

# Tim tat ca file csv trong thu muc
files = [f for f in os.listdir('.') if f.endswith('.csv')]

if not files:
    print("Khong tim thay file csv nao. Hay paste file vao cung thu muc voi lab4.py")
else:
    for file_name in files:
        print(f"\n=== DANG XU LY: {file_name} ===")
        try:
            df = pd.read_csv(file_name)
            df = df.dropna()

            # Label Encoding
            le = LabelEncoder()
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                df['encoded_label'] = le.fit_transform(df[cat_cols[0]])
                print(f"- Da ma hoa cot: {cat_cols[0]}")

            # TF-IDF
            if len(cat_cols) > 1:
                text_col = cat_cols[1]
                df['cleaned_text'] = df[text_col].apply(lambda x: " ".join(str(x).lower().split()))
                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
                print(f"- TF-IDF thanh cong! Kich thuoc: {tfidf_matrix.shape}")
        except Exception as e:
            print(f"Loi: {e}")
            print(df.head()) 
            print(df) 