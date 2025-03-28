import jieba
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
from collections import Counter

# 啟用 tqdm 進度條功能
tqdm.pandas()

# 讀取 CSV 檔案
df = pd.read_csv("all-boards.csv")

# 載入自訂詞典
jieba.load_userdict("mydict.txt")

# 刪除連接詞語
del_words_path = 'delete_words.txt'
del_words_list = []

with open(del_words_path, 'r', encoding='utf-8') as f:
    del_words_list = [line.strip() for line in f.readlines()]

# 提取版名
def extract_board(link):
    try:
        # 解析 URL，取得路徑部分
        path_parts = urlparse(link).path.split('/')
        if len(path_parts) > 2 and path_parts[1] == "bbs":
            return path_parts[2]  # 版名是第 3 個元素
    except:
        return None  # 如果解析失敗則回傳 None

# 定義分詞函數
def tokenize_title(title):
    words = list(jieba.cut(title)) # 進行分詞（預設精確模式）
    filtered_words = [word for word in words if word not in del_words_list]  # 過濾不需要的詞
    clean_words = [word.strip() for word in filtered_words if word.strip()]  # 去除空白並過濾空字串
    return ",".join(clean_words)  # 轉回字串並用 "," 分隔

df_tokenized = pd.DataFrame()
df_tokenized["board"] = df["link"].astype(str).apply(extract_board)
df_tokenized["tokenized_title"] = df["title"].astype(str).progress_apply(tokenize_title)
df_tokenized.to_csv("tokenized_titles.csv", index=False)

print("處理完成，已儲存為 tokenized_all-boards.csv")

# ---------------檢查分詞是否有問題----------------
df = pd.read_csv("tokenized_titles.csv")  # 假設已經分好詞

# 取得所有分詞
all_words = []
for row in df["tokenized_title"].dropna():  # 移除空值
    all_words.extend(row.split(","))  # 分詞是用 "," 存的

# 計算詞頻
word_freq = Counter(all_words)
# 轉成 DataFrame 方便篩選
word_freq_df = pd.DataFrame(word_freq.items(), columns=["word", "count"])
# 按詞頻排序，找出最常出現的詞
word_freq_df = word_freq_df.sort_values(by="count", ascending=False)
word_freq_df.to_csv("word_frequency.csv", index=False)
print("詞頻統計完成，請檢查 word_frequency.csv")
