####这个是适合多轮对话的版本
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name="./Qwen2-0.5B-Instruct"


class QwenAgent:
    def __init__(self,system_prompt="你是一个很聪明的智能体，能用中文回答用户的问题"):
        self.system_prompt = system_prompt
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,

        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def ask(self,messages):
        text =self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs=self.tokenizer([text],return_tensors="pt").to(self.model.device)

        generated_ids=self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
        )

        generated_ids=[output_ids[len(input_ids):] for input_ids,output_ids in zip(model_inputs.input_ids,generated_ids)]

        response=self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]
        return response

if __name__ == "__main__":
    # 实例化示例，避免被其他脚本导入时重复加载模型。
    short_memory_agent=QwenAgent()

    messages=[
        {
            "role":"system",
            "content":short_memory_agent.system_prompt
        }
    ]

    max_round=3

    user_queries=[
        "我喜欢编程，你呢",
        "你好呀，介绍一下你自己吧",
        "简单讲一下PyTorch吧",
        "还记得我喜欢什么吗"
    ]


    
