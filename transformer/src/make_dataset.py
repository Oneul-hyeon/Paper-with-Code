import sentencepiece as spm
import pandas as pd
import os

from pathlib import Path
from tqdm import tqdm

NOW_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(NOW_PATH)
DATA_PATH = os.path.join(ROOT_PATH, "data")
RAW_PATH = os.path.join(DATA_PATH, "raw")
DF_PATH = os.path.join(DATA_PATH, "df")

raw_files = [str(path) for path in Path(RAW_PATH).glob("*.xlsx")]

merged_df = pd.DataFrame(columns=["원문", "번역문"])
for raw_file in tqdm(raw_files) :
    df = pd.read_excel(raw_file)
    merged_df = pd.concat([merged_df, df[["원문", "번역문"]]], ignore_index=True)

merged_df = merged_df.rename(columns={"원문" : "source", "번역문" : "target"})
merged_df.to_csv(os.path.join(DF_PATH, "df.csv"), index=False)