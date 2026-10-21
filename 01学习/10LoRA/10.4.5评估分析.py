
prompts=[
    "你在这做什么",
    "好久没有见到你了，你最近怎么样",
    "你最近在做什么？"
]

messages=[
    [
        {"role":"system","content":"You are a helpful assistant"},
        {"role":"user","content":prompt},
    ]
    for prompt in prompts
]

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

P_model_name="../Qwen2-0.5B-Instruct"
P_model=AutoModelForCausalLM.from_pretrained(
    P_model_name,
    dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda")
tokenizer=AutoTokenizer.from_pretrained(
    P_model_name,
    trust_remote_code=True
)
tokenizer.pad_token=tokenizer.eos_token
formatted_prompts=[
    tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    for message in messages
]

P_model_inputs=tokenizer(
    formatted_prompts,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids=P_model.generate(
    input_ids=P_model_inputs.input_ids,
    attention_mask=P_model_inputs.attention_mask,
    max_new_tokens=512,
)

output_ids=generated_ids[:,P_model_inputs.input_ids.shape[1]:]
responses=tokenizer.batch_decode(
    output_ids,
    skip_special_tokens=True,
)

for i,(instruction,response) in enumerate(zip(formatted_prompts,responses)):
    print(f"instruction{i}:{instruction}")
    print(f"response{i}:{response}")


from peft import PeftModel
lora_path="../Qwen2-0.5B-Instruct-local/checkpoint-700"
lora_model=PeftModel.from_pretrained(
    P_model,
    model_id=lora_path,
)

lora_generated_ids=lora_model.generate(
    input_ids=P_model_inputs.input_ids,
    attention_mask=P_model_inputs.attention_mask,
    max_new_tokens=512,
)


lora_output_ids=lora_generated_ids[:,P_model_inputs.input_ids.shape[1]:]
lora_responses=tokenizer.batch_decode(
    lora_output_ids,
    skip_special_tokens=True,
)
for i,(instruction,response) in enumerate(zip(formatted_prompts,lora_responses)):
    print(f"instruction{i}:{instruction}")
    print(f"response{i}:{response}")
