import pandas as pd
import os
import gc
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
from tqdm import tqdm
from urllib.parse import urlparse

tqdm.pandas()

BATCH_SIZE = 5000  # 每次處理 5000 筆，根據記憶體調整
INPUT_FILE = "all-boards.csv"
OUTPUT_FILE = "tokenized_titles.csv"

def clean(sentence_ws, sentence_pos):
    short_sentence = []
    stop_pos = {'P', 'T', 'Caa', 'Cab', 'Cba', 'Cbb','PARENTHESISCATEGORY'}
    for word_ws, word_pos in zip(sentence_ws, sentence_pos):
        if word_pos not in stop_pos and len(word_ws) > 1:
            short_sentence.append(word_ws)
    return ",".join(short_sentence)

def extract_board(link):
    try:
        path_parts = urlparse(link).path.split('/')
        if len(path_parts) > 2 and path_parts[1] == "bbs":
            return path_parts[2]
    except Exception as e:
        print(f"URL parse error: {e}")  # Debug
    return None

def process_batch(batch_df, ws_driver, pos_driver):
    text = batch_df["title"].dropna().tolist()
    ws = ws_driver(text)
    pos = pos_driver(ws)

    results = []
    for i, (sentence, sentence_ws, sentence_pos) in enumerate(zip(text, ws, pos)):
        try:
            clean_text = clean(sentence_ws, sentence_pos)
            board = extract_board(batch_df.iloc[i]["link"])
            results.append({"title": clean_text, "board": board})
        except Exception as e:
            print(f"Error processing row {i}: {e}")

    return results

def main():
    df = pd.read_csv(INPUT_FILE, chunksize=BATCH_SIZE)  # 使用 chunksize 讀取，節省記憶體

    print("Initializing CKIP models...")
    ws_driver = CkipWordSegmenter(model="albert-base", device=-1)
    pos_driver = CkipPosTagger(model="albert-base", device=-1)

    first_write = not os.path.exists(OUTPUT_FILE)  # 確定是否是第一次寫入 CSV
    for i, batch_df in enumerate(df):
        results = process_batch(batch_df, ws_driver, pos_driver)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            OUTPUT_FILE, mode='a', header=first_write, index=False, encoding='utf-8-sig'
        )
        first_write = False  # 之後都不寫 header

        # 釋放記憶體
        del results_df, results
        gc.collect()

    print("處理完成，結果已保存到", OUTPUT_FILE)

if __name__ == "__main__":
    main()
