from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv


# 本脚本在 再练/ 子目录，.env 放在项目根（上一级的上一级）
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)#加载.env文件

DS_API_Key = os.getenv('DS_API_Key')
DS_Url = os.getenv("DS_Url", "https://api.deepseek.com")
'''
os.environ["X"]        # X 不存在 → 直接抛 KeyError
os.getenv("X")         # X 不存在 → 返回 None,不报错
os.getenv("X", "默认")  # 不存在 → 返回你给的默认值
'''


if not DS_API_Key:
    raise ValueError("没读到 DS_API_Key:检查 .env 是否在项目根、变量名是否拼对")


class Agent:
    def __init__(self,api_key,base_url,system_prompt='你是一个聪明的智能体,能用中文回答用户问题'):
        self.api_key=api_key
        self.base_url=base_url
        self.system_prompt=system_prompt
        #客户端
        self.client=OpenAI(api_key=self.api_key,base_url=self.base_url)

    def ask(self,question):
        #客户端通信,本质是调用deepseek的对话功能,官网有说明
        response=self.client.chat.completions.create(
            model='deepseek-flash',
            messages=[
                {'role':'system','content':self.system_prompt},
                {'role':'user','content':question}
            ],
            stream=False
        )
        return response.choices[0].message.content

    def update_system_prompt(self,system_prompt):
        self.system_prompt=system_prompt

ds=Agent(api_key=DS_API_Key,base_url=DS_Url)

answer=ds.ask('你是谁?')
print(answer)