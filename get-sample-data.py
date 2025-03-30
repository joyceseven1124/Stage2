import pandas as pd

# 讀取原始 CSV 檔案
input_file = "tokenized_titles_v2.csv"  # 這裡換成你的檔案路徑
output_file = "tokenized_titles.csv"     # 指定輸出的檔案名稱

# 讀取 CSV
df = pd.read_csv(input_file)

# 隨機抽取 5000 筆資料（如果資料不足 5000，則取全部）
sampled_df = df.sample(n=min(1030000, len(df)), random_state=42)

# 將抽取出的資料存成新的 CSV 檔案
sampled_df.to_csv(output_file, index=False)

print(f"已成功儲存 {len(sampled_df)} 筆資料到 {output_file}")