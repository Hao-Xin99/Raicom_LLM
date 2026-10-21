# -*- coding: utf-8 -*-
"""
================================================================================
 demo_client.py —— 客户端：用 requests 去调用 fastapi_demo.py 的各个接口
================================================================================

【使用顺序】
    1. 先【另开一个终端】启动服务端，并保持它在运行：
           conda activate Raicom
           python fastapi_demo.py
    2. 再【回到这个终端】运行客户端：
           conda activate Raicom
           python demo_client.py

这个脚本就是“打电话的一方”（requests），而 fastapi_demo.py 是“接电话的一方”（FastAPI）。
它把你前面问过的客户端知识点挨个走一遍：GET 带 params、POST 带 json、读状态码、
带请求头（权限）、以及 403/200 的区别。
================================================================================
"""

import requests

BASE = "http://127.0.0.1:8000"     # 127.0.0.1 = 本机，8000 = 服务端监听的端口


def show(title, resp):
    """统一打印：状态码 + 返回的 JSON（.json() 会把返回体解析回 Python dict）。"""
    print("=" * 70)
    print(title)
    print("HTTP 状态码:", resp.status_code, "  ->", end=" ")
    if resp.status_code == 200:
        print("成功 200")
    elif resp.status_code == 403:
        print("权限不足 403")
    elif resp.status_code == 404:
        print("路径不存在 404")
    elif resp.status_code == 422:
        print("参数校验失败 422")
    else:
        print("其他状态码")
    try:
        print("返回内容:", resp.json())
    except Exception:
        print("返回内容（非 JSON）:", resp.text)


# ---------- 1. GET：最简单的健康检查 ----------
show("1) GET /health（探活）",
     requests.get(f"{BASE}/health"))

# ---------- 2. GET + 查询参数 params（?room=R305） ----------
# params 会被 requests 自动拼到网址后面，变成 /tools/shelf_query?room=R305
show("2) GET /tools/shelf_query?room=R305（查询参数）",
     requests.get(f"{BASE}/tools/shelf_query", params={"room": "R305"}))

# ---------- 3. 动态路径参数：直接把值拼进网址 ----------
show("3) GET /rooms/R305（路径参数 {room_id}）",
     requests.get(f"{BASE}/rooms/R305"))

# ---------- 4. POST + JSON 请求体：OpenAI 风格对话 ----------
show("4) POST /v1/chat/completions（OpenAI 风格，json= 请求体）",
     requests.post(
         f"{BASE}/v1/chat/completions",
         json={"messages": [{"role": "user", "content": "你好"}]},
         timeout=5))

# ---------- 5. POST + JSON 请求体：多轮对话 ----------
show("5) POST /chat（多轮对话，session 1）",
     requests.post(f"{BASE}/chat",
                   json={"session_id": "s1", "text": "我想借一本书"},
                   timeout=5))

# ---------- 6. POST + Pydantic 模型参数：摘要 ----------
show("6) POST /summarize（自动校验，summary_len 是字符数）",
     requests.post(f"{BASE}/summarize",
                   json={"text": "今天图书馆二楼机房因网络故障临时关闭。", "max_len": 10},
                   timeout=5))

# ---------- 7. GET + RAG：先查一次（未命中缓存），再查一次（命中缓存） ----------
url = f"{BASE}/rag"
show("7a) GET /rag?q=借阅卡补办需要什么材料（第一次，cached=False）",
     requests.get(url, params={"q": "借阅卡补办需要什么材料"}))
show("7b) GET /rag 同一个问题（第二次，cached=True）",
     requests.get(url, params={"q": "借阅卡补办需要什么材料"}))
show("7c) GET /rag?q=明天天气（超纲 -> 拒答，阈值 0.5）",
     requests.get(url, params={"q": "明天天气怎么样"}))

# ---------- 8. 降级：服务内部出错，但状态码仍返回 200 + fallback=True ----------
show("8) GET /may_fail（降级：HTTP 200 + fallback=True）",
     requests.get(f"{BASE}/may_fail"))

# ---------- 9. 手机号脱敏（POST，正则） ----------
show("9) POST /desensitize（手机号脱敏）",
     requests.post(f"{BASE}/desensitize",
                   json={"text": "我的手机号是 13812345678，请联系我"},
                   timeout=5))

# ---------- 10. 敏感拦截：命中 -> blocked=True ----------
show("10) POST /assistant（敏感指令 -> 被拦截）",
     requests.post(f"{BASE}/assistant",
                   json={"user": "u1", "question": "请帮我删除数据库"},
                   timeout=5))

# ---------- 11. 规则路由：报修 -> repair_agent ----------
show("11) POST /assistant（报修 -> repair_agent）",
     requests.post(f"{BASE}/assistant",
                   json={"user": "u1", "question": "R305 空调坏了，报修"},
                   timeout=5))

# ---------- 12. 权限：普通用户做高危操作 -> 403；管理员带请求头 -> 放行 ----------
# headers 用来设置请求头；X-User-Role 对应服务端的 x_user_role
show("12a) POST /admin/action?action=restart，普通用户（应 403）",
     requests.post(f"{BASE}/admin/action",
                   params={"action": "restart"}))
show("12b) POST /admin/action?action=restart，带请求头 X-User-Role: admin（应 200）",
     requests.post(f"{BASE}/admin/action",
                   params={"action": "restart"},
                   headers={"X-User-Role": "admin"}))

print("=" * 70)
print("全部演示完成。可打开本目录下的 audit_demo.jsonl 查看审计日志（每行一条 JSON）。")
print("也可以在浏览器打开 http://127.0.0.1:8000/docs 自己点着测每个接口。")
