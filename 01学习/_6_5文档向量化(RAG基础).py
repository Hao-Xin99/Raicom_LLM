import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

doc1 = (
    "PolarDB for AI 是基于 PolarDB MySQL 版的数据库内分布式机器学习组件。\n"
    "它提供了通过 SQL 语句进行机器学习模型训练、评估和推理的能力，并内置分类、回归、聚类等多种算法。\n"
    "PolarDB for AI 打通了数据库和 AI 应用之间的壁垒，为数据驱动的智能应用提供了一站式的高效方案。"
)

doc2 = (
    "PolarDB for AI 将传统的数据库+AI 模式转变为一站式的智能应用。\n"
    "基于 PolarDB for AI 的推荐算法和知识图谱技术，企业可以针对不同业务场景定制个性化解决方案，\n"
    "快速完成冷启动并持续提升核心业务指标。"
)

docs = [doc1, doc2]

model=SentenceTransformer("bge-large-zh-v1.5")

embeddings = model.encode(docs)

embeddings=embeddings/np.linalg.norm(embeddings,axis=1,keepdims=True)

dim=embeddings.shape[1]
index=faiss.IndexFlatL2(dim)
index.add(embeddings)

qry="PolarDB for AI提供了哪些功能"
qry=model.encode([qry])
qry=qry/np.linalg.norm(qry,axis=1,keepdims=True)

D,I=index.search(qry,2)

for cixu,idx in enumerate(I[0]):
    score=D[0][cixu]
    print(f"相似度为{score:.4f}")
    print(f"文本内容：{docs[idx][:50]}...")


