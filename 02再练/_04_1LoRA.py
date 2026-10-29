'''记载分词器,并以bf16加载模型'''
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from pathlib import Path
#设备    
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path=str(Path(__file__).resolve().parent.parent/'本地模型'/'Qwen2-0.5B-Instruct')
tokenizer=AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)
model=AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16    #只有模型要有这个精度，因为要存权重，tokenizer分词器是不需要的
).to(device)
#把填充符设为eos_token,保证批量处理时的兼容性
tokenizer.pad_token=tokenizer.eos_token

if __name__=='__main__':
    '''数据集读取'''
    import pandas as pd
    from datasets import Dataset
    #把数据读出为pandas的DataFrame
    json_path=str(Path(__file__).resolve().parent.parent/'01学习'/'10LoRA'/'huanhuan.json')
    df=pd.read_json(json_path)
    #把df文件读做ds文件:
    ds=Dataset.from_pandas(df=df)
    '''单个样例数据处理函数,把单条数据(对话)格式化,编码化,生成模型训练所需的input_ids,attention_mask,labels'''
    def process_eg(eg:dict):
        MAX_LENGTH=256
        #先把数据变为特定格式
        instruction_part=(  #system里是固定人设,不能乱加,所以'instruction'和'input'都放user里
            '<|im_start|>system\n You are a helpful assistant.<|im_end>\n'
            f'<|im_start|>user\n{eg['instruction']}{eg['input']}<|im_end|>\n'
            '<|im_start|>assistant\n'
        )
        response_part=f'{eg['output']}<|im_end|>'

        #对指令和输出进行分词处理,千万不要加[]！！！因为后续要对两个ids拼接
        # 直接传字符串得到的是一维列表，后续直接相加拼接
        # 但要是加了[]变为二维[[token1,token2....]],相加后就变成了[[token1,...],[token1...]],不是需要的一整串token列表！！！
        instruction_ids=tokenizer(
            instruction_part,
            add_special_tokens=False #这是一个完整问答,我们自己处理了
        )['input_ids']
        response_ids=tokenizer(
            response_part,
            add_special_tokens=False
        )['input_ids']

        #处理序列得'input_ids','labels','attention_mask'
        #1. 先得到初始'input_ids','labels'
        input_ids=instruction_ids+response_ids
        labels=[-100]*len(instruction_ids)+response_ids  #把输入隐去

        #2.防止长度超出:
        if(len(input_ids)>MAX_LENGTH):
            input_ids=input_ids[:MAX_LENGTH]
            labels=labels[:MAX_LENGTH]
        
        #3.填充前,用attention_mask记录有效部分
        attention_mask=[1]*len(input_ids)

    ##### 后面的几步是用来填充的,但deepseek说可以不要,交给后面的库DataCollator,还能省空间和步骤
        #4.进行填充
        #pad_len=MAX_LENGTH-len(input_ids)

        #input_ids+=[tokenizer.pad_token_id]*pad_len    #对input_ids填充token_id,一定要是token_id（数字），不是token这个字符
        #labels+=[-100]*pad_len  #填充部分不计算loss
        #attention_mask+=[0]*pad_len #填充部分是无效的,而且由于用eos_token代替pad_token,模型是分不清谁是真正的结尾,这时候attention_mask就能告诉:假结束是0

        #5.把处理好的这些序列返回
        return {'input_ids':input_ids,
                'labels':labels,
                'attention_mask':attention_mask
        }



    '''对整个数据集进行处理,map把原来的ds表格每一行处理后,加入列,还需要我们手动去掉新列,生成的还是一个dataset'''
    ds_processed=ds.map(
        function=process_eg,
        remove_columns=ds.column_names
    )

    '''LoRA模块配置,注入原模型'''
    from peft import LoraConfig,get_peft_model,TaskType
    lora_config=LoraConfig(
        task_type=TaskType.CAUSAL_LM,#因果语言构建
        target_modules=['q_proj','v_proj','k_proj','o_proj'],#注入模型
        inference_mode=False,#训练模式,非推理
        r=8,    #LoRA的秩
        lora_alpha=32,
        lora_dropout=0.05
    )
    #根据这些配置,把原来模型改装一下
    lora_model=get_peft_model(model=model,peft_config=lora_config)
    #get_peft_model*()会根据配置,把原来的权重全部冻住
    save_path=str(Path(__file__).resolve().parent/'04LoRA_quan')
    '''开始微调训练'''
    from transformers import TrainingArguments,Trainer,DataCollatorForSeq2Seq
    #配置训练参数
    traing_args=TrainingArguments(
        output_dir=save_path,  #权重保存地址

        per_device_train_batch_size=8,#一个卡一次吃的样本数
        gradient_accumulation_steps=4,#完成几次训练更新一次权重.

        num_train_epochs=3,#整个数据集训练几次

        learning_rate=1e-4,#学习率
        
        logging_steps=50,#状态记录，记下来这个状态的loss，学习率等
        save_steps=100,#把目前模型学到的权重保存下来

        save_on_each_node=True,
        gradient_checkpointing=False
    )
    #过程:一次训练吃8个样本,进行一次前向算loss,反向算梯度,梯度攒着,进行4次训练后在取梯度平均更新参数.等效于一次训练吃32个样本
        #数据集一共有3279个数据,一轮要训练3279/32=117,一共3轮351次训练
        #每50次训练终端打印一次状态,每100次训练保存一次权重


    #DataCollator把样本批量整理为模型可直接输入的张量
    #可以自动padding,所以前面的数据处理函数可以省略对齐
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True
    )

    #构建Trainer实例
    trainer=Trainer(
        model=lora_model,
        args=traing_args,
        data_collator=data_collator,
        train_dataset=ds_processed,
    )
    #启动训练
    trainer.train()




