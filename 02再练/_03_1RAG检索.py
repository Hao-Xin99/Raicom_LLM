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
            dtype='auto'
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
        '''
        text是apply_chat_template返回的一句纯文本(分段是因为\n):
        <|im_start|>system\n
        你是助手<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
        注意最后一行 assistant:add_generation_prompt=True 的作用就是在末尾补上“该你说了”的起始标记，模型会从该标记后继续生成。
        '''
       
        inputs=self.tokenizer(
            [text],
            return_tensors='pt'
        ).to(self.model.device)
        #inputs是tokenizer函数返回的是一个类似字典{'input_ids':[batch,token_all],'attention_mask':[batch,token_all]}
        #这里tokenizer如果text不加[]，返回的就是一个一维列表[token1,token2,...],如果加[]，返回的是二维[[token1,token2,...]]，加【】后就是（batch,token_total）的形状了
        #这里需要加【】，因为我们要的就是二维，符合后续输入的形状要求
        generated=self.model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        #generate()返回的是list,[batch,总token]
        generated=[output_ids[len(input_ids):]for input_ids,output_ids in zip(inputs.input_ids,generated)]

        response=self.tokenizer.batch_decode(generated,skip_special_tokens=True)[0]

        return response


#防止import文件时重读全文件
if __name__ == '__main__':  

    agent=QwenAgent()
    '''文档预处理:就是对文档切片为文档列表,与手工切基本一样,这里用库切一个'''
    from langchain_text_splitters import CharacterTextSplitter
    text_path=Path(__file__).resolve().parent.parent/'01学习'/'zhuzhiwu.txt'
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
    #单位化.
    #faiss.normalize_L2(embedded)
    embedded_norm=embedded/np.linalg.norm(embedded,axis=1,keepdims=True)
    #构建Faiss向量索引
    dim=embedded_norm.shape[1]
    faiss_index=faiss.IndexFlatIP(dim)
    faiss_index.add(embedded_norm)

    #保存向量索引和原始文档,方便后续读取,文档与向量索引对应
    faiss.write_index(faiss_index,'zhuzhiwu.faiss_idx')

    with open('zhuzhiwu_chunks.pkl','wb')as f:      #wb:以二进制写入
        pickle.dump(chunks,f)



    #测试是否可以成功索引
    query='烛之武是怎么退秦师的?'
    query_em=Encode_model.encode([query])
    faiss.normalize_L2(query_em)
    D,I=faiss_index.search(query_em,k=2)
    for rank,idx in enumerate(I[0]):
        print(f'Rank:{rank},分数:{D[0][rank]:.4f},片段:\n{chunks[idx]}')


