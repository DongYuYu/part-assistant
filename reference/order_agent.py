"""
最小的「查訂單 + 退款」agent 示範。

重點看三件事：
1. 回應的 content 是一塊一塊的 block，每塊都有 type（text / tool_use / thinking ...）
2. 程式只看 stop_reason 和 block.type 就知道下一步要做什麼
3. 我們能限制的是「工具」和「輸出格式」，不是 content type 本身

執行前：pip install anthropic，並設定 ANTHROPIC_API_KEY（或用 `ant auth login`）。
"""

import json

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5-5"

# ---- 假的訂單資料庫 ---------------------------------------------------------
ORDERS = {
    "A123": {"金額": 2300, "狀態": "已到貨", "購買天數": 5},
    "B456": {"金額": 8800, "狀態": "已到貨", "購買天數": 2},
}

# ---- 工具定義：模型「只能」呼叫這裡列出的工具，參數也只能照 schema ----------
TOOLS = [
    {
        "name": "lookup_order",
        "description": "用訂單編號查詢訂單的金額、狀態、購買天數。",
        "strict": True,  # 保證 input 一定符合下面的 schema
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "訂單編號，例如 A123"},
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "issue_refund",
        "description": "對訂單發起退款。",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "integer", "description": "退款金額（元）"},
                # enum：模型只能從這幾個選項裡挑
                "reason": {"type": "string", "enum": ["瑕疵", "不想要了", "寄錯商品"]},
            },
            "required": ["order_id", "amount", "reason"],
            "additionalProperties": False,
        },
    },
]

SYSTEM = "你是電商客服。先查訂單再處理退款。回覆客戶時用繁體中文、簡短有禮。"

# ---- 原始 JSON 檢視 ----------------------------------------------------------
SHOW_RAW = True        # 印出每次呼叫送出的 request 和收到的 response
SHOW_FULL_TOOLS = False  # 工具定義每次都一樣，預設只印名稱；改 True 看完整 schema
call_count = 0


def _jsonable(obj):
    """SDK 回傳的 block 是物件，轉成 dict 才能用 json.dumps 印出來。"""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"無法轉成 JSON：{type(obj)}")


def pretty(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=_jsonable)


def call_api(**params):
    """包住 client.beta.messages.create，前後把原始資料印出來。"""
    global call_count
    call_count += 1

    if SHOW_RAW:
        shown = dict(params)
        if "tools" in shown and not SHOW_FULL_TOOLS:
            shown["tools"] = [f"<{t['name']} 的定義，略>" for t in shown["tools"]]
        print(f"\n{'>' * 20} 第 {call_count} 次 API 呼叫：REQUEST {'>' * 20}")
        print(pretty(shown))

    response = client.beta.messages.create(**params)

    if SHOW_RAW:
        print(f"\n{'<' * 20} 第 {call_count} 次 API 呼叫：RESPONSE {'<' * 20}")
        print(pretty(response.to_dict()))

    return response


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """真的去執行工具。回傳 (結果文字, 是否出錯)。"""
    if name == "lookup_order":
        order = ORDERS.get(args["order_id"])
        if order is None:
            return f"找不到訂單 {args['order_id']}", True
        return json.dumps(order, ensure_ascii=False), False

    if name == "issue_refund":
        # 權限檢查寫在程式裡，不靠模型自律
        if args["amount"] > 5000:
            return "金額超過 5000 元，系統拒絕自動退款，需轉真人審核。", True
        return f"已退款 {args['amount']} 元（原因：{args['reason']}）", False

    return f"未知的工具 {name}", True


def show(response) -> None:
    """把每次回應的外殼印出來，看清楚格式。"""
    print(f"\n=== stop_reason: {response.stop_reason} ===")
    for block in response.content:
        if block.type == "text":
            print(f"[text] {block.text}")
        elif block.type == "tool_use":
            print(f"[tool_use] id={block.id} name={block.name} "
                  f"input={json.dumps(block.input, ensure_ascii=False)}")
        else:
            # 例如 thinking：Opus 5.5 預設會思考，但內容預設不顯示
            print(f"[{block.type}]")


def agent(customer_message: str, max_turns: int = 6) -> None:
    print(f"\n##### 客戶：{customer_message}")
    messages = [{"role": "user", "content": customer_message}]

    for _ in range(max_turns):  # 停止條件 1：最多跑幾輪
        response = call_api(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "low"},  # 簡單任務，少花一點 token
            # 若模型因安全分類拒絕，伺服器自動改用合適的備援模型
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        show(response)
        # 整包 content 放回紀錄（包含 thinking 區塊），不要只放文字
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":  # 停止條件 2：模型說完了（或被截斷/拒絕）
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output, is_error = run_tool(block.name, block.input)
                print(f"  -> 程式執行 {block.name}：{output}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # 對應是哪一個請求
                    "content": output,
                    "is_error": is_error,
                })
        messages.append({"role": "user", "content": results})  # 所有結果一次送回

    print("超過最大輪數，轉真人處理。")


# ---- 示範二：完全不用工具，只要一份固定格式的資料 -----------------------------
CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["退款", "物流", "帳號", "其他"]},
        "urgent": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["category", "urgent", "summary"],
    "additionalProperties": False,
}


def classify(email: str) -> dict:
    response = call_api(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": f"分類這封客服信：\n{email}"}],
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": CLASSIFY_SCHEMA},
        },
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)  # 格式有保證，可以直接解析


if __name__ == "__main__":
    agent("訂單 A123 的杯子有裂痕，我想退款。")
    agent("訂單 B456 我不想要了，請全額退款。")  # 會撞到程式的 5000 元限制

    print("\n##### 示範二：Structured Outputs")
    print(classify("我上週買的耳機到現在還沒收到，明天就要出國了！！"))

    print(f"\n總共呼叫 API {call_count} 次")
