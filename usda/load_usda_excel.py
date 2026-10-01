import pandas as pd
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent
DATA_PATH= BASE_DIR/"data"/"usda_foods.xlsx"
usda_df= pd.read_excel(DATA_PATH,header=1)

usda_df['Food code']=usda_df['Food code'].astype(int)

print(usda_df.columns)