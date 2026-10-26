#实测微调后模型效果
from transformers import AutoTokenizer
from peft import PeftModel
from pathlib import Path
from _04_1LoRA import model_path,model,device #分词器依旧使用原模型
Lora_id_path=str(Path(__file__).resolve().parent/'04LoRA_quan'/'checkpoint-351')

model_xun=PeftModel.from_pretrained(
    model=model,
    model_id=Lora_id_path
).to(device)
tokenizer=AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)
tokenizer.pad_token=tokenizer.eos_token

query='你在这里做什么!'
message=[
    {'role':'system','content':'You are a helpful assistant'},
    {'role':'user','content':query}
]

inputs=tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt=True
)
#输入编码并且搬入gpu
input_ids=tokenizer(
    [inputs],
    return_tensors='pt'
).to(device)

response_ids=model_xun.generate(
   # **input_ids,
    input_ids=input_ids.input_ids,
    attention_mask=input_ids.attention_mask,
    max_new_tokens=512
)
#response_ids形状(batch,token)

response_ids=[output_id[len(input_id):]for output_id,input_id in zip(response_ids,input_ids.input_ids)]
#这里这个response_ids是二维列表,只是只有一个元素

response=tokenizer.batch_decode(response_ids,skip_special_tokens=True)[0]#取第一个问题的答案

print(response)