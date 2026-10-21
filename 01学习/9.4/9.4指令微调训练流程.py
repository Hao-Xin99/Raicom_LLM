
#模型选择与参数设置
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device ="cuda" if torch.cuda.is_available() else "cpu"
model_path="./disrilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

tokenizer.pad_token = tokenizer.eos_token

#配置训练参数
from transformers import TrainingArguments
training_args=TrainingArguments(
    output_dir="./9.4output",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    report_to="none"
)

#3.加载并预处理数据
from datasets import load_dataset

alpaca_datasets=load_dataset("tatsu-lab/alpaca")

def preprocess_func(examples):
    prompt=[
        f"###指令：{instruction}\n###输入：{input_text}\n###回答：{output}"
        for instruction,input_text,output in zip(examples["instruction"],examples["input"],examples["output"])
    ]
    tokenized_examples=tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    tokenized_examples["labels"]=[
        [token_id if token_id !=tokenizer.pad_token_id else -100 for token_id in seq]
        for  seq in tokenized_examples["input_ids"]
    ]
    return tokenized_examples

tokenized_dataset=alpaca_datasets.map(
    preprocess_func,
    batched=True,
    remove_columns=alpaca_datasets["train"].column_names,
)

##步骤4：启动微调训练
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=None
)

Result=trainer.train()
print(Result.training_loss)



















