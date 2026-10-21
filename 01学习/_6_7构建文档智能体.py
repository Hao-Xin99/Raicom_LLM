from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import faiss
import torch


from _6_0定义智能体模板 import QAgent

embedding_model=SentenceTransformer("./bge-large-zh-v1.5")

with open("chunks.pkl","rb") as f:
    doc_chunks=pickle.load(f)

index=faiss.read_index("My_index.faiss")






def ask_agent(qry,agent,top_k=3):
    qry_ebd=embedding_model.encode([qry])
    qry_ebd=qry_ebd/np.linalg.norm(qry_ebd,axis=1,keepdims=True)

    D,I=index.search(qry_ebd,top_k)
    received_chunks=[doc_chunks[i] for i in I[0]]
    "\n\n".join(received_chunks)

    prompt=(
        "你是一个问答助手，请基于以下资料回答用户问题：\n"
        "参考资料:\n"
        F"{received_chunks}\n\n"
        f"用户问题：{qry}\n"
        "请用清晰，完整的中文回答："
    )
    message=[
        {"role":"system","content":agent.system_prompt},
        {"role":"user","content":prompt},
    ]

    response=agent.ask(message)
    return response

doc_agent=QAgent()

question="烛之武是怎劝退秦王的"

answer=ask_agent(question,doc_agent)
print(answer)







