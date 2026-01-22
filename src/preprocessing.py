# src/preprocessing.py
import pandas as pd


def preprocess(df):
    df = df.copy()

    def clean_text(text: str) -> str:
        if not isinstance(text, str) or pd.isna(text):
            return ''
        text = text.lower().strip()
        return text

    text_columns = ['title', 'overview', 'genres', 'keywords']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
            df[col] = df[col].apply(clean_text)
        else:
            df[col] = ''

    df['combined_text'] = (
            df['title'] + ' ' +
            df['overview'] + ' ' +
            df['genres'] + ' ' +
            df['keywords']
    ).str.strip()



    return df