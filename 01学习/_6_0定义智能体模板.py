
from transformers import AutoTokenizer, AutoModelForCausalLM

class QAgent:
    def __init__(self,system_prompt="你是一个聪明的智能体。"):
        #初始化分词器和大模型
        self.tokenizer =AutoTokenizer.from_pretrained(
            "./Qwen2-0.5B-Instruct",
            trust_remote_code=True,
        )
        self.model=AutoModelForCausalLM.from_pretrained(
            "./Qwen2-0.5B-Instruct",
            trust_remote_code=True,
            device_map='auto',
            torch_dtype='auto'
        )
        #推理模式
        self.model.eval()
        #系统提示词
        self.system_prompt=system_prompt
        #对话历史
        self.messages=[
            {"role":"system","content":system_prompt},
        ]

    def ask(self,question):
        self.messages.append({"role":"user","content":question})
        #构建对话模板,方便向量化
        text=self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        #把模板化的输入编码,并放入gpu
        model_input=self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        generated=self.model.generate(
            **model_input,
            max_new_tokens=2048,
            truncation=True,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )

        generated_ids=[output_ids[len(input_ids):] for input_ids,output_ids in zip(model_input.input_ids,generated)]

        response=self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]

        return response


