# -*- coding: utf-8 -*-
"""
================================================================================
 fastapi_demo.py  ——  FastAPI 速通教学（对照 2026 睿抗「大模型及智能体应用」样题）
================================================================================

这个文件把比赛里会用到的 Web 用法【全部集中在一个文件、一个端口(:8000)】里演示。
真实项目里它们会被拆到四层（8700 模型层 / 9500 工具层 / 8100 应用层 / 8200 智能体层），
这里为了方便你一次性看懂，先用一个进程演示【每一种写法】；每段都标注了它对应的
样题任务/知识点。

【怎么运行】
    1. 打开终端，激活环境：        conda activate Raicom
    2. 进到本文件所在目录后运行：  python fastapi_demo.py
    3. 看到 “Uvicorn running on http://127.0.0.1:8000” 即启动成功（Ctrl+C 停止）

【启动后在浏览器依次访问】
    http://127.0.0.1:8000/health                 健康检查（GET，最简单）
    http://127.0.0.1:8000/tools/shelf_query      工具层风格 GET
    http://127.0.0.1:8000/docs                   FastAPI 自动生成的交互文档（可直接点着测）
    （POST 类接口用同目录的 demo_client.py 或 /docs 页面测试）

【知识点回顾（前面问过的概念，都在本文件里有对应代码）】
    进程/端口/HTTP  ->  本文件被 uvicorn 跑起来后，占住 8000 端口
    @app.get/post   ->  装饰器：把下面的函数“登记”到对应路径（对应任务2/5/9）
    路径/方法/参数   ->  GET 用查询参数 ?k=v；POST 用请求体 JSON；{} 是动态路径段
    async def/await ->  读取请求体 req.json() 是“等待型”操作，所以用 async def + await
    返回 dict        ->  FastAPI 自动转成 JSON
    JSONResponse     ->  自己控制状态码/字段（降级时仍返回 200，对应任务7）
    HTTPException    ->  主动返回错误状态码（如 403 权限不足，对应任务10）
    多轮/缓存/审计   ->  对应任务5、7、10
================================================================================
"""

# ---------------- 第 0 部分：导入（用到什么 import 什么） ----------------
import re               # 正则：任务3 手机号脱敏
import json             # json.dumps：任务9 写 JSONL 日志
import time             # 时间戳、缓存有效期 TTL
import hashlib          # md5：任务7 缓存键
from datetime import datetime

import requests         # 客户端库：跨进程调用别的服务（智能体层调工具层）
from fastapi import FastAPI, Request, Query, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel      # 请求体模型：自动校验类型（任务5/6 常用）
import uvicorn                     # 真正把 app 跑起来、监听端口的服务器


# ---------------- 第 1 部分：创建应用（对应每个 .py 里的 app = FastAPI()） ----------------
# 后面所有 @app.xxx 都登记在这个 app 上；title 会显示在 /docs 文档里
app = FastAPI(title="图书馆 AI 平台 · FastAPI 教学 Demo")


# ---------------- 第 2 部分：内存里的“假数据库/全局状态”（重启即清空，仅用于演示） ----------------
HISTORY = {}        # 多轮对话历史：{session_id: [消息,...]}      对应任务5
CACHE = {}          # 缓存：{md5(问题): (写入时间戳, 结果)}        对应任务7
AUDIT_LOG_PATH = "audit_demo.jsonl"   # 审计日志文件（每行一条 JSON）对应任务10

# 工具层的假数据（真实项目由 kb_tools_server.py :9500 提供）
ROOMS = {
    "R305": {"room": "R305", "status": "空调停机", "level": "P1"},
    "R201": {"room": "R201", "status": "运行正常", "level": "NA"},
}

# 安全规则（对应任务10）
SENSITIVE = ["删除数据库", "关机", "重启服务器", "清除日志"]
REPAIR_KEYWORDS = ["报修", "故障", "空调", "门禁", "网络"]

# 工具注册表：工具名 -> 它的完整网址（对应任务9.1 TOOLS）
# 真实项目端口是 9500；本 demo 全部在 8000，便于你一个人跑通
TOOLS = {
    "shelf_query":    "http://127.0.0.1:8000/tools/shelf_query",
    "log_query":      "http://127.0.0.1:8000/tools/log_query",
    "kb_query":       "http://127.0.0.1:8000/tools/kb_query",
    "resource_query": "http://127.0.0.1:8000/tools/resource_query",
}


# ---------------- 通用小工具函数 ----------------
def now_iso() -> str:
    """生成 ISO 格式时间字符串，审计日志里的 timestamp 字段用它。"""
    return datetime.now().isoformat(timespec="seconds")


def append_jsonl(path: str, record: dict) -> None:
    """把一条 dict 以【追加模式 "a"】写成 JSONL 日志（每行一条 JSON）。
    对应任务9.4：open(..., "a") + json.dumps(..., ensure_ascii=False)。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def mock_llm(text: str) -> str:
    """假的“大模型生成”：不真正调用模型，只把输入拼一句返回。
    对应降级链最末端的【规则兜底 mock】；真实项目这里会 HTTP 调模型层/Ollama。"""
    return f"【模拟模型回答】你刚才说的是：{text}"


# =================================================================================
# 第 3 部分：工具层风格 —— GET 接口 + 查询参数 Query（对应 kb_tools_server.py，4 个 GET）
# =================================================================================
@app.get("/health")
def health():
    """健康检查：几乎每层都有，访问它就回“我还活着”，用于探活。"""
    return {"status": "ok", "service": "fastapi_demo", "time": now_iso()}


@app.get("/tools/shelf_query")
def shelf_query(room: str = Query("R305")):
    """查询座位/设备状态。
    用法：/tools/shelf_query?room=R305
    Query("R305") 表示查询参数 room 的默认值是 R305（对应任务9.5 的 room=Query("R305")）。"""
    return ROOMS.get(room, {"room": room, "status": "未知房间", "level": "NA"})


@app.get("/tools/log_query")
def log_query(service: str = Query("rfid")):
    """查询服务日志。默认服务代号 rfid（对应任务9.5 空79）。"""
    return {"service": service, "log": f"[{service}] 10:02 发生一次离线告警"}


@app.get("/tools/kb_query")
def kb_query(q: str = Query("借阅卡补办")):
    """知识库检索（这里给一条写死的命中结果，对应任务9.5 空81 的“上门处理”）。"""
    return {"q": q, "result": "命中：维修/制卡人员 2 个工作日内上门处理。"}


@app.get("/tools/resource_query")
def resource_query(db: str = Query("cnki")):
    """电子资源并发查询（对应任务9.5：cnki / 128 / 120 / 已满）。"""
    return {"db": db, "concurrent": 128, "limit": 120, "status": "已满"}


@app.get("/rooms/{room_id}")
def get_room(room_id: str):
    """【动态路径参数】演示：花括号 {room_id} 是“占位符”，能匹配任意值并自动传进函数。
    用法：/rooms/R305  -> room_id="R305"；/rooms/abc -> room_id="abc"。
    注意它和上面的固定路径 /tools/shelf_query 不冲突，是两种不同写法。"""
    info = ROOMS.get(room_id)
    return {"room_id": room_id, "info": info if info else "没有这个房间"}


# =================================================================================
# 第 4 部分：模型层风格 —— OpenAI 兼容接口 /v1/chat/completions（对应任务2，空8~11）
# =================================================================================
@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    """OpenAI 风格的对话接口。
    - async def：因为下面要 await 读取请求体（网络等待）；
    - await req.json()：把 POST 发来的 JSON 请求体读出来（这就是必须 async 的原因）；
    - 请求体是标准格式 {"messages":[{"role":"user","content":"..."}]}；
    - 返回也模仿 OpenAI 的 choices 结构。"""
    body = await req.json()
    messages = body["messages"]
    # 从对话列表里取最后一条 user 消息
    last_user = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user = m["content"]
            break
    answer = mock_llm(last_user)
    return {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer},
             "finish_reason": "stop"}
        ],
    }


# =================================================================================
# 第 5 部分：应用层 —— 多轮对话 / 通知摘要（对应任务5）
# =================================================================================
@app.post("/chat")
async def chat(req: Request):
    """多轮对话：用 HISTORY 按 session_id 记住上下文（对应任务5.1）。
    每轮追加两条消息：user 提问 + assistant 回复。"""
    body = await req.json()
    sid, text = body["session_id"], body["text"]

    # setdefault：键不存在就先用 [] 建好，再取出来（对应空26 的 HISTORY.setdefault("sid", [])）
    history = HISTORY.setdefault(sid, [])
    history.append({"role": "user", "content": text})

    answer = mock_llm(text)
    history.append({"role": "assistant", "content": answer})

    return {"session_id": sid, "reply": answer}


class SummarizeIn(BaseModel):
    """请求体模型（Pydantic）：FastAPI 会自动把 POST 的 JSON 解析成它并校验类型。
    text 必须是字符串，max_len 不传时默认 100。"""
    text: str
    max_len: int = 100


@app.post("/summarize")
def summarize(data: SummarizeIn):
    """通知摘要：把文本压到 max_len 字以内（对应任务5.2/5.4）。
    注意这里用的是【普通 def】，FastAPI 同样支持（会放到线程池执行）。
    summary_len 字段用内置函数 len() 统计字符数（对应空35/59）。"""
    s = "".join(data.text.split())     # 去掉空白，简单处理
    if len(s) > data.max_len:
        s = s[:data.max_len]
    return {"summary": s, "summary_len": len(s)}


# =================================================================================
# 第 6 部分：RAG 检索增强 + 缓存 + 阈值拒答（对应任务6、7）
# =================================================================================
def fake_retrieve(question: str, top_k: int = 3):
    """假检索：真实项目用 bge 嵌入 + faiss；这里用关键词模拟相似度。
    返回 (分数列表, 命中的政策文本)。top_k=3 对应任务6.3（空42）。"""
    if any(k in question for k in ["借阅卡", "补办", "学生证", "照片"]):
        return [0.92, 0.71, 0.55], "补办借阅卡需携带本人学生证和一张一寸照片，到总服务台办理。"
    return [0.21, 0.12, 0.05], ""     # 超纲问题：最高分很低


@app.get("/rag")
def rag(q: str = Query(...), clear: bool = False):
    """知识增强问答：
    - Query(...) 表示 q 是【必填】查询参数，不传会自动报 422；
    - 缓存：用问题的 md5 做键，TTL 内直接返回（对应空49~51），命中时 cached=True（空60）；
    - 阈值拒答：最高分 < 0.5 就友好拒答（对应空47）。
    访问：/rag?q=借阅卡补办需要什么材料      （命中）
          /rag?q=明天天气怎么样            （拒答）
    """
    threshold = 0.5
    key = hashlib.md5(q.encode("utf-8")).hexdigest()   # 空49：hashlib.md5(...).hexdigest()
    ttl = 60                                          # 空50：有效期 60 秒

    # 1) 先看缓存
    if key in CACHE and time.time() - CACHE[key][0] < ttl:
        return {"answer": CACHE[key][1], "cached": True, "scores": []}

    # 2) 没命中则“检索”
    scores, policy = fake_retrieve(q, top_k=3)
    best_score = float(scores[0])
    if best_score < threshold:                        # 空47：低于阈值则拒答
        answer = "未检索到相关政策资料，请咨询图书馆总服务台。"
    else:
        answer = policy

    CACHE[key] = (time.time(), answer)               # 空51：写入缓存
    return {"answer": answer, "cached": False, "scores": scores, "best_score": best_score}


@app.post("/cache/clear")
def cache_clear():
    """清空缓存（对应图里的 /cache/clear）。"""
    n = len(CACHE)
    CACHE.clear()
    return {"cleared": n}


# =================================================================================
# 第 7 部分：降级 —— 出错也返回友好提示 + HTTP 200（对应任务7.2/7.5，空52/53/58）
# =================================================================================
@app.get("/may_fail")
def may_fail():
    """演示模型服务异常时的降级：用 JSONResponse 显式构造响应，
    带 fallback=True 标记，同时把状态码保持为 200（不让前端看到 500 白屏）。"""
    try:
        return 1 / 0                       # 故意制造一个异常
    except Exception:
        return JSONResponse(
            content={"reply": "系统繁忙，请稍后重试，或致电图书馆总服务台。",
                     "fallback": True},    # 空52：降级标记字段
            status_code=200,               # 空53：即使失败也返回 200
        )


# =================================================================================
# 第 8 部分：智能体层 —— 敏感拦截 + 规则路由 + 权限 + 审计（对应任务10）
# =================================================================================
def route(question: str) -> str:
    """关键词规则路由（对应空83~85）：命中报修关键词 -> repair_agent，否则 visitor_agent。"""
    if any(k in question for k in REPAIR_KEYWORDS):
        return "repair_agent"
    return "visitor_agent"


@app.post("/assistant")
async def assistant(req: Request, x_user_role: str = Header(default="user")):
    """工单统一入口：先做敏感拦截，再做规则路由，并用 agent 字段标明实际应答者（空86）。
    Header(default="user") 表示从请求头 X-User-Role 读取角色，缺省为普通用户。"""
    body = await req.json()
    question = body.get("question", "")
    user = body.get("user", "anonymous")

    # 1) 敏感指令拦截（空87~89）：命中直接挡下，并写审计
    if any(s in question for s in SENSITIVE):
        append_jsonl(AUDIT_LOG_PATH,
                     {"timestamp": now_iso(), "user": user,
                      "action": "assistant", "result": "blocked"})
        return {"reply": "检测到敏感指令，已拦截并记录。", "agent": None, "blocked": True}

    # 2) 规则路由
    agent = route(question)

    # 3) 报修类演示“先查设备状态”：这里同进程直接调用 shelf_query()。
    #    真实项目这里用 call_tool() 经 HTTP 调工具层 :9500（见文件末尾说明）。
    extra = shelf_query("R305") if agent == "repair_agent" else {}

    append_jsonl(AUDIT_LOG_PATH,
                 {"timestamp": now_iso(), "user": user,
                  "action": "assistant", "result": agent})
    reply = f"已为您接入【{agent}】。"
    if extra:
        reply += f"当前设备状态：{extra['status']}（{extra['level']}）"
    return {"reply": reply, "agent": agent, "blocked": False}


@app.post("/admin/action")
def admin_action(action: str = Query(...),
                 x_user_role: str = Header(default="user")):
    """高权限操作（重启设备/关闭工单）：仅管理员可做，否则抛 403（对应空90~92）。
    HTTPException 会让 FastAPI 自动返回对应状态码。
    测试：不带请求头调用 -> 403；带请求头 X-User-Role: admin -> 放行。"""
    if action in ["restart", "close_ticket"] and x_user_role != "admin":
        raise HTTPException(status_code=403, detail="权限不足")
    append_jsonl(AUDIT_LOG_PATH,
                 {"timestamp": now_iso(), "user": x_user_role,
                  "action": action, "result": "ok"})
    return {"ok": True, "action": action, "operator": x_user_role}


# =================================================================================
# 第 9 部分：数据脱敏（对应任务3.1 正则）+ 跨进程调用工具（对应任务9）
# =================================================================================
PHONE_RE = re.compile(r"1[3-9]\d{9}")   # 空15：大陆手机号，1 开头、第二位 3-9、共 11 位


@app.post("/desensitize")
async def desensitize(req: Request):
    """把文本里的手机号用掩码替换（空16 的 PHONE_RE.sub(掩码, 文本)）。"""
    body = await req.json()
    text = body.get("text", "")
    masked = PHONE_RE.sub("***********", text)
    return {"raw": text, "masked": masked}


def call_tool(name: str, args: dict):
    """跨进程调用工具层（对应任务9.1）：requests.get + params + timeout=5（空71/72）。
    注意：这是给【客户端或另一个服务】用的，不要在本服务唯一的工作线程里同步调它自己，
    否则可能死锁；真实项目里它调用的是另一个端口（9500），由 demo_client.py 演示。"""
    return requests.get(TOOLS[name], params=args, timeout=5).json()


# =================================================================================
# 第 10 部分：启动服务（uvicorn 才是真正监听端口的服务器）
# =================================================================================
if __name__ == "__main__":
    # 演示用 8000，避免和真实四层端口（8700/9500/8100/8200）冲突
    uvicorn.run(app, host="127.0.0.1", port=8000)
