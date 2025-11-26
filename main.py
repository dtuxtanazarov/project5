import json
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


data = open('dataset.json', 'r', encoding='utf8')
dataset = json.load(data)


model = SentenceTransformer("")

contexts = [d["context"] for d in data]
questions = [d["question"] for d in data]
answers = [d["answer"] for d in data]

context_emb = model.encode(contexts)
question_emb = model.encode(questions)
answer_emb = model.encode(answers)


combined_emb = np.concatenate([context_emb, question_emb, answer_emb], axis=0)
print("Asl o'lcham:", combined_emb.shape)


pca = PCA(n_components=100)
reduced = pca.fit_transform(combined_emb)

print("Qisqartirilgan o'lcham:", reduced.shape)  # (3, 100)

df = pd.DataFrame(data)
for i in range(reduced.shape[1]):
    df[f"pca_{i+1}"] = reduced[:, i]

df.to_csv("uzbek_QA_PCA.csv", index=False, encoding="utf-8-sig")

print("✅ Natija saqlandi: uzbek_QA_PCA.csv")

