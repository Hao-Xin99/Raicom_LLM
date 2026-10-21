from openai import OpenAI
Key="sk-a2edc4bf42974ce9a3a1543239d3a30e"
Base_Url="https://api.deepseek.com"

def get_response(
        system_prompt:str,
        user_prompt:str,
        model:str="deepseek-v4-pro",
        temperature:float=0.5,
        top_p:float=0.9,
        max_tokens:int=2048,
)->str:
    client=OpenAI(
    api_key=Key,
    base_url=Base_Url,
    )
    message=[{"role":"system","content":system_prompt},
         {"role":"user","content":user_prompt},
        ]
    response=client.chat.completions.create(
        model=model,
        messages=message,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
)
    return response.choices[0].message.content

system_prompt="你是一个助教，能温柔地回答同学问题"
user_prompt=("最能体现大模型语言生成能力的是什么?"
             "A. 基于上下文和指令概率分布，自回归式地预测并输出后续文本"
             "B. 利用倒排索引技术在本地大规模语料库中执行 Sematic Search"
             "C. 使用传统的正则表达式（Regex）匹配文本中的结构化命名实体"
             "D. 通过有监督分类器将文本分配到预定义的标签体系中"
             )
response=get_response(system_prompt,user_prompt)

print(response)


