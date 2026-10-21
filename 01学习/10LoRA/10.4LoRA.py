##1.加载模型
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name="../Qwen2-0.5B-Instruct"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)

tokenizer=AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

tokenizer.pad_token=tokenizer.eos_token
##2.读取数据
import pandas as pd
from datasets import Dataset

df=pd.read_json("./huanhuan.json")
ds=Dataset.from_pandas(df)
##3.数据预处理
def preprocess(example):
    MAX_LENGTH=512
    instruction_parts=(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}{example['input']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    response_parts=(
        f"{example['output']}<|im_end|>\n"
    )
    instruction_ids=tokenizer(
        instruction_parts,
        add_special_tokens=False,
    )["input_ids"]
    response_ids=tokenizer(
        response_parts,
        add_special_tokens=False,
    )["input_ids"]

    input_ids=instruction_ids+response_ids
    labels=[-100]*len(instruction_ids)+response_ids

    if len(input_ids)>MAX_LENGTH:
        input_ids=input_ids[:MAX_LENGTH]
        labels=labels[:MAX_LENGTH]

    attention_mask=[1]*len(input_ids)

    paddings=MAX_LENGTH-len(input_ids)
    input_ids+=[tokenizer.pad_token_id]*paddings
    labels+=[-100]*paddings
    attention_mask+=[0]*paddings

    return {
    "input_ids": input_ids,
    "labels": labels,
    "attention_mask": attention_mask,
     }
tokenizer_id=ds.map(
    preprocess,
    remove_columns=ds.column_names
)

##LoRA配置
from peft import LoraConfig,get_peft_model,TaskType
lora_config=LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj','k_proj','v_proj','o_proj'],
    inference_mode=False,
    r=2,
    lora_alpha=8,
    lora_dropout=0.5,
)
lora_model=get_peft_model(model,lora_config)

model_save_path=f"{model_name}-local"
lora_model.print_trainable_parameters()

#开始训练LoRA
from transformers import Trainer,TrainingArguments,DataCollatorForSeq2Seq

training_args=TrainingArguments(
    output_dir=model_save_path,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=50,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=False,
)
data_collator=DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenizer_id,
)

trainer.train()

