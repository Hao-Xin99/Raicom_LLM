from openai import OpenAI


class QwenAgent:
    """连本地 Ollama 里的 qwen3.5，接口和原来 transformers 版保持一致。"""

    def __init__(self, system_prompt="你是一个聪明的智能体。"):
        # Ollama 不校验 key，随便填；base_url 指向本地 11434 的 OpenAI 兼容口
        self.client = OpenAI(
            api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
        )
        self.model = "qwen3.5:latest"   # 和 `ollama list` 里的名字逐字一致
        self.system_prompt = system_prompt

    def ask(self, messages):
        """
        兼容两种调用方式：
          1. _6_7 里传完整 message 列表：[{"role":"system",...},{"role":"user",...}]
          2. 旧代码传单条字符串："你好"
        返回纯字符串。
        """
        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": messages},
            ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )
        return resp.choices[0].message.content


# 自测：直接跑这个文件能看到效果
if __name__ == "__main__":
    bot = QwenAgent()
    print(bot.ask("你是谁？用一句话介绍自己。"))
