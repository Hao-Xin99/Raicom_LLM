from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from pathlib import Path
from fastapi import FastAPI,Request
import uvicorn
from fastapi.responses import PlainTextResponse,JSONResponse

model_path=str(Path(__file__).resolve().parent.parent/'本地模型'/'Qwen2-0.5B-Instruct')
device='cuda' if torch.cuda.is_available() else 'cpu'

app=FastAPI(description='这是一个单次问答大模型')


#模型提前加载好
model=AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16
    ).to(device)
model.eval()#开启推理模式

tokenizer=AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True
    )


@app.post('/v1/chat/completions')
async def ask_llm(req:Request):
    content=await req.json()
    #content是{'role':'user','content':query}
    
    messages=[
        {'role':'system','content':'You are a helpful assistant'},
    ]
    messages.append(content)

    #把问题输入进行格式化
    inputs=tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    #进行编码,input_ids是个字典:{'input_ids':形状(batch,tokens)列表,'attention_masks':同上形状列表}
    input_ids=tokenizer(
        [inputs],
        return_tensors='pt'
    ).to(device)
    #输出
    output_ids=model.generate(
        **input_ids,
        #后面是输出配置:
        max_new_tokens=256,
        do_sample=True,
        top_k=3,
        top_p=0.3,
        temperature=0.6
    )
    #对输出进行裁剪
    output_ids=[out_id[len(inp_id):] for out_id,inp_id in zip(output_ids,input_ids.input_ids)]
    #对输出ids进行解码
    response=tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]
    #这里原本用纯文本了,不对,要用json
    #return PlainTextResponse(response)
    return JSONResponse(
        content={'res':response},
        status_code=200,
    )

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)


#curl -s -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type:application/json" -d "{\"role\":\"user\",\"content\":\"你是谁\"}" -o "C:\Users\17740\Desktop\Raicom\03样题\submission\llm.json" -w "%{http_code}"