# import os
# from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

#工具类会引用的东西
from langchain.tools import Tool
import requests
from langchain_community.utilities import SerpAPIWrapper
from datetime import datetime

#能思考使用啥工具的智能体
from langchain.agents import AgentExecutor,create_react_agent
from langchain import PromptTemplate

import os
from pathlib import Path
from dotenv import load_dotenv



env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)
DS_API_Key = os.getenv('DS_API_Key')
DS_Url = os.getenv("DS_Url", "https://api.deepseek.com")

# from langchain_experimental.plan_and_execute.executors import agent_executor

#初始化智能体
llm=ChatOpenAI(
    model="deepseek-flash",
    base_url=DS_Url,
    api_key=DS_API_Key,
    temperature=0.7,
    max_tokens=500,
)
#记忆模块
memory=ConversationBufferWindowMemory(

    memory_key="chat_history",
    return_messages=True,
)

####工具调用模块

#计算器
def calculator(expression:str)->str:
    try:
        result=eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

calc_tool=Tool(
    name="calculator",
    func=calculator,
    description="用于数学计算。输入数学表达式",
)

#获取当前时间
def get_current_time(query:str)->str:
    now=datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
get_time_tool=Tool(
    name="current_time",
    func=get_current_time,
    description="获取当前日期的函数"
)

#网页搜索（百度）
search=SerpAPIWrapper(
    params={"engine":"baidu"},
    serpapi_api_key="08757678789f1b7f9215530cd43845158114bfdbc7c5d52c4221e6177d45fff4"
)

def web_search(query:str)->str:
    return search.run(query=query)

web_search_tool=Tool(
    name="web_search",
    func=web_search,
    description="用于网页查询"
)

#网页内容抓取
def get_web_content(url:str)->str:
    try:
        response=requests.get(url,timeout=10)
        return response.text[:2000]
    except Exception as e:
        return f"无法访问网页{str(e)}"
get_web_content_tool=Tool(
    name="get_web_content",
    func=get_web_content,
    description="获取指定URL的网页内容"
)

def generate_summary(text:str)->str:
    prompt=f"请对以下内容生成一个简洁的摘要:\n\n{text}"
    return llm.predict(prompt)
summary_tool=Tool(
    name="summary",
    func=generate_summary,
    description='生成文本摘要'
)

def query_memory(query:str)->str:
    history=memory.load_memory_variables({})['chat_history']
    history_text="\n".join([f"{x.type}:{x.content}"for x in history])
    prompt=f"根据以下对话历史，回答用户问题:\n\n{history_text}\n\n问题：{query}"
    return llm.predict(prompt)

query_tool=Tool(
    name="MemoryQuery",
    func=query_memory,
    description="记忆查询"
)

#工具集
tools=[
    calc_tool,
    get_time_tool,
    web_search_tool,
    get_web_content_tool,
    summary_tool,
    query_tool,
    ]

#系统提示词
prompt_template="""
你是一个有用的AI助手。可以使用一下工具回答问题：
{tools}
可用工具名称：
{tool_names}
请用一下格式回答:
Question: 用户的问题
Thought: 我需要思考如何回答这个问题
Action: 要使用的工具名称
Action Input: 工具的输入参数
Observation: 工具返回的结果
...（这个思考/行动/观察的过程可以重复多次）
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答

对话历史：
{chat_history}

Question:{input}
Thought:{agent_scratchpad}
"""

prompt=PromptTemplate(
    template=prompt_template,
    input_variables=["tools","tool_names","chat_history","input","agent_scratchpad"]

)

#创建智能体
agent=create_react_agent(llm,tools,prompt)

agent_executor=AgentExecutor(
    agent=agent,
    memory=memory,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=60,
)

####实例化


#交流函数
def chat_with_agent(user_input:str)->str:
    try:
        response=agent_executor.invoke({"input":user_input})
        return response["output"]
    except Exception as e:
        return f"抱歉，出错了{str(e)}"


#创建简易的交互界面
def interact():
    # 打印欢迎信息和功能介绍
    print("\n" + "=" * 50)
    print("🌐 网页搜索与信息汇总助手")
    print("✨ 功能包括: ")
    print("    - 实时网络搜索")
    print("    - 网页内容抓取")
    print("    - 自动文本摘要")
    print("    - 对话记忆功能")
    print("💡 输入 'quit' 退出程序")
    print("=" * 50)
    while True:
        try:
            user_input=input("\n您：").strip()

            if user_input.lower() in ["quit","exit"]:
                print("再见")
                break
            if not user_input:
                continue
            print("智能体正在思考")
            response=chat_with_agent(user_input)
            print(f"AI:{response}")
        except KeyboardInterrupt:
            print("\n再见啦")
            break
        except Exception as e:
            print(f"系统出错了{str(e)}")
            break
interact()


