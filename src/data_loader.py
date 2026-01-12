import pandas as pd


def load_movie_data(file_path: str) -> pd.DataFrame:
    try: 
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    
    if df.empty:
        raise ValueError(f"The file '{file_path}' is empty.")
    
    required_columns=["id", "title", "genres", "overview"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The required column '{col}' is missing from the data.")
        
    return df


