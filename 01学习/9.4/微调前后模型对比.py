from transformers import pipeline
import torch
agent_yuan=pipeline(
    "text-generation",
    model="D:\PyCharm\PycharmProjects\PythonProject\disrilgpt2",
    tokenizer="D:\PyCharm\PycharmProjects\PythonProject\disrilgpt2",
    device=0 if torch.cuda.is_available() else -1,
)

agent_xun=pipeline(
    "text-generation",
    model="D:\PyCharm\PycharmProjects\PythonProject\9.4output\checkpoint-4200",
    tokenizer="D:\PyCharm\PycharmProjects\PythonProject\disrilgpt2",
    device=0 if torch.cuda.is_available() else -1,
)

prompt = (
    "###指令：Summarize the following article in no more than two sentences.\n"
    "###输入：The rapid advancement of artificial intelligence has led to breakthroughs in natural language processing, computer vision, and robotics, enabling new applications across healthcare, finance, transport, and education. These developments are improving efficiency, reducing human error, and opening up opportunities for personalized services and innovative research.\n"
    "###回答："
)

gene_1=agent_yuan(
    prompt,
    max_length=100,
    num_return_sequences=1,
)[0]["generated_text"]
print(f"原始模型：{gene_1}")

gene_2=agent_xun(
    prompt,
    max_length=100,
    num_return_sequences=1,
)[0]["generated_text"]
print(f"训练模型：{gene_2}")
