from IPython.core.display_functions import display
from openai import OpenAI

client=OpenAI(
    api_key="sk-cf8eb977505d4296980c143425d787a2",
    base_url="https://api.deepseek.com",
)

def ask(system_prompt,user_prompt,messages=None):
    if messages is None:
        messages=[]
    final_message=[{"role":"system","content":system_prompt}]
    final_message+=messages
    final_message.append({"role":"user","content":user_prompt})

    response=client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=final_message,
        stream=False
    )
    return response.choices[0].message.content

planner_prompt="你是一个Planner Agent，擅长分析文章主题并罗列大纲"
writer_prompt="你是一个Writer Agent，擅长根据要点生成高质量的中文段落"

context={
    "gole":"撰写一篇300字左右的文章，主题是《简要概括复变函数与积分变换的内容与思想》"
}
planner_output=ask(
    planner_prompt,
    F"请帮我把下面的主题拆解成3~5个段落要点：{context['gole']}",
)
print(f"拆解结果：\n{planner_output}\n\n")

sub_tasks=[
    line for line in planner_output.split("\n")
    if line.strip()and line.strip()[0].isdigit()
]
paras=[]
for sub_task in sub_tasks:
    para=ask(writer_prompt,f"请你根据以下要点写一个段落：{sub_task}")
    paras.append(para)

draft="\n\n".join(paras)

from IPython.display import Markdown

display(Markdown(draft))