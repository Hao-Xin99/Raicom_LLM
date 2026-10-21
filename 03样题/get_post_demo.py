# -*- coding: utf-8 -*-
# 最小演示：GET vs POST 的区别，以及 Request 对象里到底装了什么
# 运行：在 Raicom 环境里执行  python get_post_demo.py
import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import json

app = FastAPI()

# ---------- ① GET：数据在 URL 的 ? 后面，函数直接声明参数 ----------
@app.get("/hello")
def hello(name: str):
    return {"你好": name}

# ---------- ② POST：数据在请求体 JSON 里，用 req.json() 读 ----------
@app.post("/chat")
async def chat(req: Request):
    data = await req.json()          # 把请求体 JSON 解析成字典
    return {"收到的JSON": data}

# ---------- ③ 把 Request 对象里的关键信息回显出来，让你看它长啥样 ----------
@app.post("/inspect")
async def inspect(req: Request):
    return {
        "method":     req.method,            # 请求方法：POST / GET
        "完整URL":     str(req.url),
        "路径path":    req.url.path,
        "请求头headers": dict(req.headers),   # 里面有 content-type、角色等
        "请求体body":  await req.json(),      # POST 发来的 JSON
    }


client = TestClient(app)

print("=" * 55)
print("【GET 演示】数据拼在 URL 的 ? 后面")
r1 = client.get("/hello", params={"name": "小明"})
print("  请求地址 =", r1.request.url, "   <- 地址栏直接能看到 name=小明")
print("  返回     =", r1.json())

print("=" * 55)
print("【POST 演示】数据藏在请求体里，URL 上什么都看不到")
r2 = client.post("/chat", json={"msg": "你好模型"})
print("  请求地址 =", r2.request.url, "   <- URL 干干净净，数据在 body")
print("  返回     =", r2.json())

print("=" * 55)
print("【Request 对象里到底装了什么？发个 POST 让它自己报出来】")
r3 = client.post("/inspect",
                 json={"question": "开馆时间"},
                 headers={"X-User-Role": "admin"})
print(json.dumps(r3.json(), ensure_ascii=False, indent=2))
