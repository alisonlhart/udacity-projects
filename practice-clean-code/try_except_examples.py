import pandas as pd

def read_csv(csv_pth):
    try:
        df = pd.read_csv(csv_pth)
        print(df.head())
        return df
    except FileNotFoundError:
        print("Couldn't find the damn thing.")
        
data = read_csv("")