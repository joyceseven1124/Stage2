from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np

df = pd.read_csv('tokenized_titles.csv' ,header=None, dtype=str)
df_clean = [[word.strip() for word in str(line).split(",")] for line in df[0]]

# 轉為 DataFrame 使用 .sample()
df_clean_df = pd.DataFrame(df_clean)
df_sample = df_clean_df.sample(frac=1, random_state=42).reset_index(drop=True)
# 將因為sample中產生的None清理乾淨
df_sample = df_sample.map(lambda x: x if x is not None else '')
df_sample = df_sample.apply(lambda x: [i for i in x if i != ''], axis=1)

# 將資料分成兩部分
test_count = 1001
assert len(df) > test_count, "資料不足，請確保 `df` 至少有 1000 筆資料"
test_df = df_sample.sample(n=test_count, random_state=42)
train_df = df_sample.drop(test_df.index).reset_index(drop=True)
# 創建 TaggedDocument ,將字詞上tag
print('Tagged Documents Ready')
tagged_data = [TaggedDocument(words=row, tags=[index]) for index, row in enumerate(train_df.values)]


print('Start Training')
model = Doc2Vec(vector_size=30,window=2, min_count=2, epochs=60, workers=4)
# 建立詞彙表
model.build_vocab(tagged_data)
print("Training model...")
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)


print('Test Similarity')
second_similarities = []
self_similarities = []
min_similarity = 0.8
MODEL_FILENAME = "doc2vec_model"
# 測試模型
for doc_id, test_doc in enumerate(test_df):
    if doc_id % 100 == 0:
        print('目前測試進度',doc_id)
    test_doc = test_df.iloc[doc_id]
    inferred_vector = model.infer_vector(test_doc)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))  # 取得所有文檔的相似度
    # 第一
    self_similarity = sims[0][1]
    self_similarities.append(self_similarity)
    # 將第二名的排名進行紀錄
    second_similarity = sims[1][1]
    second_similarities.append(second_similarity)

# 計算平均第二相似度
avg_self_similarity = np.mean(self_similarities)
print(f"self similarity: {avg_self_similarity:.2%}")
avg_second_similarity = np.mean(second_similarities)
print(f"second self similarity: {avg_second_similarity:.2%}")

# 如果平均第二相似度超過 80%，則保存模型
if avg_second_similarity >= min_similarity:
    model.save(MODEL_FILENAME)
    print(f"模型已保存至 {MODEL_FILENAME}")
else:
    print("模型未達標準，未保存。")