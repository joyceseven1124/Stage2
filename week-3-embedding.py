from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np

df = pd.read_csv('tokenized_titles.csv' ,header=None, dtype=str)
df_clean = [[word.strip() for word in str(line).split(",")] for line in df[0]]

# 轉為 DataFrame 使用 .sample()
df_clean_df = pd.DataFrame(df_clean)
df_sample = df_clean_df.sample(frac=1, random_state=42).drop_duplicates().reset_index(drop=True)
# 將因為sample中產生的None清理乾淨
df_sample = df_sample.map(lambda x: x if x is not None else '')
df_sample = df_sample.apply(lambda x: [i for i in x if i != ''], axis=1)

# 將資料分成兩部分
test_count = 1001
assert len(df) > test_count, "資料不足，請確保a `df` 至少有 1000 筆資料"
test_df = df_sample.sample(n=test_count, random_state=42)
train_df = df_sample

# test_df.to_csv('test_data.csv', header=False)
# train_df.to_csv('train_data.csv', header=False)
# 創建 TaggedDocument ,將字詞上tag
print('Tagged Documents Ready')
tagged_data = [TaggedDocument(words=row, tags=[index]) for index, row in enumerate(train_df.values)]


print('Start Training')
model = Doc2Vec(vector_size=30, min_count=2, epochs=150, alpha=0.025, min_alpha=0.00025, dm=1, workers=4)
# 建立詞彙表
model.build_vocab(tagged_data)
print("Training model...")
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)


print('Test Similarity')
self_in_top1 = 0
self_in_top2 = 0
total_tests = len(test_df)
min_similarity = 0.8
MODEL_FILENAME = "doc2vec_model"

# 測試模型
for doc_id, test_doc in enumerate(test_df):
    if doc_id % 100 == 0:
        print('目前測試進度',doc_id)
    test_doc = test_df.iloc[doc_id]
    inferred_vector = model.infer_vector(test_doc)
    sims = model.dv.most_similar([inferred_vector], topn=2) # 取得所有文檔的相似度

    top1, top1_sim = sims[0]  # 第一名
    top2, top2_sim = sims[1]  # 第二名

    if top1 in test_df.index:
        self_in_top1 += 1
    if top1 in test_df.index or top2 in test_df.index:
        self_in_top2 += 1
# 計算平均第二相似度
p1 = self_in_top1 / total_tests
p2 = self_in_top2 / total_tests
print(f"self similarity: {p1:.2%}")
print(f"second self similarity: {p2:.2%}")

# 如果平均第二相似度超過 80%，則保存模型
if p2 >= min_similarity:
    model.save(MODEL_FILENAME)
    print(f"模型已保存至 {MODEL_FILENAME}")
else:
    print("模型未達標準，未保存。")