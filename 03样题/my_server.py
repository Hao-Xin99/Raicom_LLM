#####我写一个服务器

from fastapi import FastAPI,Request
import uvicorn
from fastapi.responses import JSONResponse,PlainTextResponse
app=FastAPI()

@app.get('/name')
def name(name:str):
    return PlainTextResponse(f'你好,{name}')

@app.post('/information')
async def infor(req:Request):
    body=await req.json()
    if not {'name','ID'}<=body.keys():  #这里是集合包含的关系
        return JSONResponse(content='error:字段不全',status_code=400)
    return {
        'name':body['name'],
        'ID':body['ID'],
        'class':body['class']
    }
if __name__ == "__main__":
    uvicorn.run(app=app,host='127.0.0.1',port=8700)
        
    