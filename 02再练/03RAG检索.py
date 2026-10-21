#构建一个基础的大模型
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

# 基于本脚本文件的位置定位模型，不依赖“从哪个目录运行”
# 本文件在 Raicom/再练/，模型在 Raicom/本地模型/
MODEL_PATH = Path(__file__).resolve().parent.parent / "本地模型" / "Qwen2-0.5B-Instruct"

device='cuda' if torch.cuda.is_available() else 'cpu'

#定义一个基础模型
class QwenAgent():
    def __init__(self,system_prompt='你是一个聪明的大模型,可以回答用户的问题'):
        #定义分词器和模型本体
        self.tokenizer=AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            trust_remote_code=True,
            )
        self.model=AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            trust_remote_code=True,
            #device_map='auto',    #这个参数必须有accelerate库搭配,也可以直接像现在用to.(device平替)
            torch_dtype='auto'
            ).to(device)
        self.system_prompt=system_prompt
        self.messages=[
                    {"role":"system","content":system_prompt},
                ]
    def ask(self,query):

        self.messages.append({"role":"user","content":query})
        
        text=self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs=self.tokenizer(
            [text],
            return_tensors='pt'
        ).to(self.model.device)

        generated=self.model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )

        generated=[output_ids[len(input_ids):]for input_ids,output_ids in zip(inputs.input_ids,generated)]

        response=self.tokenizer.batch_decode(generated,skip_special_tokens=True)[0]

        return response

agent=QwenAgent()


'''文档预处理:就是对文档切片为文档列表,与手工切基本一样,这里用库切一个'''
from langchain_text_splitters import CharacterTextSplitter
text_path=Path(__file__).resolve().parent.parent/'学习'/'zhuzhiwu.txt'
#加载文本
with open(str(text_path),'r',encoding='utf-8')as f:
    text=f.read()
#文本分割器
text_splitter=CharacterTextSplitter(separator='\n',chunk_size=50,chunk_overlap=10)

chunks=text_splitter.split_text(text)
'''
for idx,chunk in enumerate(chunks):
    print(f'Chunk{idx}:{chunk}')
'''

'''文本处理完,将文本转化为向量构建检索索引,这是RAG的重要步骤'''
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

#加载中文处理模型,用它来进行编码
C_model_path=Path(__file__).resolve().parent.parent/'本地模型'/'bge-large-zh-v1.5'
Encode_model=SentenceTransformer(str(C_model_path))
#对chunks的每一段进行编码
embedded=Encode_model.encode(chunks)
#单位化
embedded_norm=embedded/np.linalg.norm(embedded,axis=1,keepdims=True)
#构建Faiss向量索引
dim=embedded_norm.shape[1]
faiss_index=faiss.IndexFlatIP(dim)
faiss_index.add(embedded_norm)

#保存向量索引和原始文档,方便后续读取,文档与向量索引对应
faiss.write_index(faiss_index,'zhuzhiwu.faiss_idx')

with open('zhuzhiwu_chunks.pkl','wb')as f:
    pickle.dump(chunks,f)


query='烛之武是怎么退秦师的?'
query_em=Encode_model.encode([query])
query_em=query_em/np.linalg.norm(query_em,axis=1,keepdims=True)

D,I=faiss_index.search(query_em,k=2)

for rank,idx in enumerate(I[0]):
    print(f'Rank:{rank},分数:{D[0][rank]:.4f},片段:\n{chunks[idx]}')
