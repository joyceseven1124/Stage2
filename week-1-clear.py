import os
import pandas as pd
import re
from tqdm import tqdm
import csv  

directory = "./raw"
if not os.path.exists(directory ):
    os.makedirs(directory )
prefix_data = {}

def process_title(title):
    if pd.isna(title):
        return ''
    # 轉換為小寫
    title = title.lower()
    # 移除前後空格
    title = title.strip()
    # 移除開頭的 "Re:" 或 "Fw:"
    title = re.sub(r'^(re: re:|re:|fw:|re: r: |r:)\s*', '', title, flags=re.IGNORECASE)

    return title

for filename in tqdm(os.listdir(directory)):
  if filename.endswith('.csv'):
    prefix = filename.split('2025')[0].strip('_')
    # 讀取檔案位置和執行檔案不同地方
    file_path = os.path.abspath(os.path.join(directory, filename))
    df = pd.read_csv(file_path, dtype=str)
    # 處理標題
    if 'title' in df.columns:
        df['title'] = df['title'].apply(process_title)
        df['title'] = df['title'].apply(process_title)

    if prefix in prefix_data:
      prefix_data[prefix] = pd.concat([prefix_data[prefix], df], ignore_index=True)
    else:
      prefix_data[prefix] = df

if prefix_data:
  for prefix,data in prefix_data.items():
    folder = './cleaned'
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_filename = f"{prefix}_combined.csv"
    file_path = os.path.join(folder, output_filename)
    # data.to_csv(file_path, index=False, encoding="utf-8", quoting=csv.QUOTE_NONE, escapechar='\\')
    data.to_csv(file_path, index=False, encoding="utf-8")
    print(f"已保存 {output_filename}")


input_folder = "./cleaned"
output_file = "./all-boards.csv"

all_data = pd.DataFrame()

for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path, quotechar='"', dtype=str)
        all_data = pd.concat([all_data, df], ignore_index=True)

all_data.to_csv(output_file, index=False, encoding="utf-8")

print(f"全部版面整合完畢 {output_file}")  