from openai import OpenAI

# ===== 本地 Ollama 配置 =====
# Ollama 不校验密钥，api_key 随便填一个非空字符串就行；真正重要的是下面两个
OLLAMA_API_KEY = "ollama"
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

# ⚠️ 模型名必须和命令行 `ollama list` 里显示的逐字一致（通常是 名字:标签，例如 qwen2.5:7b）
# 注意：这是 Ollama 自己下载的模型，不能直接填你“本地模型”文件夹里的路径
MODEL_NAME = "qwen3.5"


class Agent:
    def __init__(self, api_key, base_url, model,
                 system_prompt="你是一个聪明的智能体，请用中文回答用户问题。"):
        # 客户端：指向本地 Ollama，而不是 DeepSeek 服务器
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # 自己维护整段对话历史。
        # 原理：每次请求都是“无状态”的——服务器不记得上一句说了什么，
        # 所以必须我们自己把 system + 所有历史 user/assistant 消息一起发过去。
        self.messages = [{"role": "system", "content": system_prompt}]

    def ask(self, question: str) -> str:
        # 1) 把用户这一句追加到历史末尾
        self.messages.append({"role": "user", "content": question})

        # 2) 把“整段历史”一次性发给本地模型
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=False,
        )
        answer = response.choices[0].message.content

        # 3) 把模型这一句也存进历史，下一轮它才“记得”上文
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def set_system_prompt(self, system_prompt: str):
        """替换开场白（历史里第一条就是 system）。"""
        self.messages[0] = {"role": "system", "content": system_prompt}

    def reset(self):
        """清空除开场白外的所有对话记忆，重新开始一段。"""
        del self.messages[1:]


if __name__ == "__main__":
    bot = Agent(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL, model=MODEL_NAME)
    try:
        print(bot.ask("你是谁？"))
        # 多轮记忆测试：第二句若能复述出上一句，说明历史维护成功
        print(bot.ask("我上一句问了什么？"))
    except Exception as e:
        print("连接本地模型失败：", e)
        print("请依次检查：")
        print("  1) Ollama 已在运行（命令行敲 `ollama serve`，或它已在后台驻留）；")
        print("  2) 已经 `ollama pull <模型名>` 下载过模型；")
        print("  3) 上面的 MODEL_NAME 和 `ollama list` 的输出逐字一致。")
