import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
#模型路径   等价一句:model_path='./本地模型/Qwen2-0.5B-Instruct'
import os
from pathlib import Path
from dotenv import load_dotenv
env_path=Path(__file__).resolve().parent.parent/'.env'
load_dotenv(env_path)
Model_Path=os.environ['Path_Qwen2-0.5B-Instruct']
#分词器和模型加载
tokenizer=AutoTokenizer.from_pretrained(Model_Path)
model=AutoModelForCausalLM.from_pretrained(Model_Path)
#迁移模型至gpu
device='cuda'if torch.cuda.is_available() else 'cpu'
model=model.to(device)
#文本续写任务()
from transformers import pipeline

text_generator=pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=device)

prompt='从前有个小猪,在猪圈吃饭饭'
generated=text_generator(
    prompt,
    max_new_tokens=200, 

    do_sample=True,  #do_sample决定是否随机采样
    temperature=1.5,
    top_k=50,
    top_p=0.5,
    num_return_sequences=2,  #一次返回几条独立生成结果
    repetition_penalty=1.2,  #防重复,对使用过的词降权,=1不惩罚,1.1惩罚
)[1]['generated_text']
#text_generator()返回一个回答列表,数量取决于num_return_sequences,[0]表示取第一个
#取出来的是字典,['generated_text']是找键值对
print(generated)

