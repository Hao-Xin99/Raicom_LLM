#加载与索引构建时相同的模型
from pathlib import Path
from sentence_transformers import SentenceTransformer
C_model_path=Path(__file__).resolve().parent.parent/'本地模型'/'bge-large-zh-v1.5'
C_model=SentenceTransformer(str(C_model_path))
'''要把Path对象转化回str'''
#加载当时构建的向量空间
import faiss
import  numpy as np
Index_Path = Path(__file__).resolve().parent/'zhuzhiwu.faiss_idx'
'''
Index=faiss.read_index(str(Index_Path)) #faiss.read_index不接受中文路径!!!
'''
with open(Index_Path, 'rb') as f:                     # Python 的 open 认中文路径
    Index = faiss.deserialize_index(np.frombuffer(f.read(), dtype='uint8'))

#加载与向量空间匹配的文本块:
import pickle
chunks_Path=Path(__file__).resolve().parent/'zhuzhiwu_chunks.pkl'
with open(str(chunks_Path),'rb')as f:
    chunks=pickle.load(f)
'''pickle.load()加载的是打开的文件对象,即open后的对象'''
#构建RAG_Agent的问答
from _03_1RAG检索 import QwenAgent
def Ask_By_RAGagent(query:str,agent:QwenAgent,top_k:int=3):
    #先把问题向量化,并且归一化
    #编码encode的是一个列表,可能不止一个问题,所以后面才会有D[0],I[0]
    query_embedding=C_model.encode([query])
    faiss.normalize_L2(query_embedding)
    #RAG检索,要用检索空间
    _,Idx =Index.search(query_embedding,k=top_k)
    #返回的时(Distance相似度,Index索引),两个的形状都是(问题数,k),Index是表示这条在向量对应chunks(文本形式)的索引位置
    #问题数看query_embedding,如果是一维就是1,如果是二维数组,形状(问题数,一个问题编码的数),所以这里都取[0]
    chosen=Idx[0][0]    #这里只选了第一个
    text_add=chunks[chosen]
    #根据检索到的生成增强型提示词
    query_rag=f'''
    这是检索到的相关知识:{text_add}\n
    请你据此回答用户的问题:{query}
    '''

    return agent.ask(query_rag)

Agent=QwenAgent()

query='为什么刚开始烛之武对王请求会有推辞?'
answr=Ask_By_RAGagent(query=query,agent=Agent)
print(answr)
    




