# MCP server：同一組工具，接到任何 MCP host

一句話：**把 agent 的 4 個工具包成 MCP server；模型和對話由 host（Claude Code、Claude Desktop）負責，server 只執行工具，所以 agent 迴圈裡的安全規則要搬進 server。已在真的 Claude Code 實測過：高嚴重度工單會跳出表單。**

程式：`src/part_assistant/mcp_server.py`。測試：`tests/test_mcp_server.py`、`tests/test_audit.py`、`tests/test_access.py`。
安裝到各種 host 的步驟：見 [README](../README.md#接到你的-mcp-host)。

## 目錄

- [一、分工：誰在思考，誰在做事](#一分工誰在思考誰在做事)
- [二、設定和使用流程](#二設定和使用流程)
- [三、規則從 agent 搬進 server](#三規則從-agent-搬進-server)
- [四、實測（Claude Code）](#四實測claude-code)
- [五、限制](#五限制)
- [六、常見問題（Q&A）](#六常見問題qa)

## 一、分工：誰在思考，誰在做事

```
你 ──問題──▶ host（Claude Code：模型 + 對話迴圈，用你的訂閱或 API key）
                 │  需要資料時呼叫工具（stdio 上的 JSON-RPC）
                 ▼
            part-assistant MCP server（只執行工具，不呼叫任何語言模型）
                 ├─ search_docs   → 檢索（bge-m3 + reranker，本機 GPU），依權限和現行版本過濾
                 ├─ lookup_part   → SQLite
                 ├─ check_stock   → SQLite
                 └─ create_ticket → SQLite；高嚴重度先請 host 跳表單給人確認
                 每次工具呼叫 → audit_log（channel = mcp）
```

和我們自己的 agent 比較，工具是同一組（`Tools` 類別），差別只在誰負責思考：

| | 自己的 agent（`agent.py`） | MCP |
|---|---|---|
| 模型 | 我們的程式呼叫 Claude API | host 內建 |
| API key | 要，按 token 計費 | **不需要**；費用算在 host（訂閱或 host 自己的 key） |
| 看得到使用者的問題嗎 | 看得到，稽核紀錄記整題 | **看不到**，只收到工具呼叫 |
| 安全規則由誰執行 | agent 迴圈（harness） | server |

所以稽核紀錄在 MCP 下是「每次工具呼叫一筆」，不是「每個問題一筆」。

## 二、設定和使用流程

### 1. 準備資料（只做一次）

```bash
uv sync
uv run python -m part_assistant.download
uv run python -m part_assistant.seed
uv run python -m part_assistant.chunk
```

### 2. 登記到 host（只做一次）

```bash
mkdir -p ~/mcp-test && cd ~/mcp-test
claude mcp add part-assistant -- uv run --directory /absolute/path/to/part-assistant python -m part_assistant.mcp_server
```

- `--` 之後是**以後要執行**的啟動指令，現在只是存起來，不會執行。
- `--directory`：uv 先切換到專案資料夾再啟動，程式和 `data/` 的相對路徑才找得到。
- 範圍：預設 `local`（只有你、只在這個資料夾）；`--scope user`（只有你、任何資料夾）；專案附的 `.mcp.json` 是 `project` 範圍（第一次要核准）。

**為什麼建議在空資料夾用**：在專案資料夾裡，Claude Code 可以直接讀 `data/internal/` 的機密文件和資料庫，繞過 server 的權限過濾。真實使用者的電腦上也不會有原始碼。

### 3. 使用

```bash
cd ~/mcp-test && PART_ASSISTANT_USER=alice claude
```

1. **你不用自己啟動 server**：server 走 stdio，Claude Code 啟動時會照登記的指令把它當子程式開起來，`/exit` 時一起結束。
2. `/mcp` 確認 `part-assistant` 已連線、有 4 個工具。
3. 直接用中文問問題。第一次呼叫工具時，Claude Code 會問你是否允許。
4. 高嚴重度工單會跳出表單，要填名字並同意。

`PART_ASSISTANT_USER` 決定身分：稽核紀錄記誰，`search_docs` 也只搜這個人能看的文件（見 [access.md](access.md)）。沒設定的話用作業系統帳號，不在 `users` 表裡就只能看公開文件。

### 4. 事後查紀錄（免費）

```bash
uv run python -m part_assistant.audit --user alice
```

### 5. 除錯和移除

- 手動執行啟動指令：`uv run --directory /absolute/path/to/part-assistant python -m part_assistant.mcp_server`。資料沒準備好會印出缺了什麼；**停住不動代表正常**（在等 stdin 的訊息），Ctrl+C 結束。
- `claude --debug` 可以看到 server 的錯誤訊息（host 畫面上通常只有 `Error executing tool`）。
- 移除：`claude mcp remove part-assistant`。

## 三、規則從 agent 搬進 server

host 的模型不是我們控制的，所以「不能靠模型遵守」的規則都在 server 裡：

| 規則 | agent 裡怎麼做 | server 裡怎麼做 |
|---|---|---|
| idempotency key | agent 迴圈依「對話 + 料號 + 嚴重度」產生 | server 產生；一個 server 程序 = 一次對話（stdio 每次 host 對話啟動一個程序）。工具 schema 沒有 key 欄位，模型無法指定 |
| 高嚴重度要人確認 | agent 停下來，CLI 問人 | **elicitation**：server 請 host 跳表單，人同意並填名字才寫入 |
| host 不支援表單 | — | 呼叫失敗，**不寫入**（fail closed） |
| 拒絕、取消、沒填名字 | 不寫入 | 不寫入，回傳 `not_created` 和原因，也記進稽核紀錄 |
| 重試不重複問人 | — | 問人之前先查這個 key 是否已有工單；有就直接回傳，不跳表單 |
| 使用者身分 | `--user` | server 啟動時固定（`PART_ASSISTANT_USER`），**不是工具參數**，模型不能自稱別人 |
| 稽核 | 每題一筆 | 每次工具呼叫一筆 |

其他實作細節：

- **問人的動作放在 resolver**（`Resolve(ask_a_person)`）：在工具本體之前執行，所以本體只會在已經有答案時才跑。resolver 必須沒有副作用，因為它可能被執行不只一次。
- **檢索模型延後載入**：第一次 `search_docs` 才載入 bge-m3 和 reranker，server 在一秒內就能啟動，host 不會因為等太久判定它失敗。
- **全部是 `async def`**：在 server 的 event loop 執行緒上跑，也就是開 SQLite 連線的那個執行緒（sqlite3 不允許跨執行緒）。代價是一次搜尋會卡住 loop 約 0.5 秒，單一使用者的 stdio 可以接受。
- **stdout 是協定專用的**：server 裡絕對不能 `print()`，log 一律寫到 stderr。
- **工具描述只有一份**：MCP 工具的描述直接取自 `TOOL_SCHEMAS`，和 agent 用的完全相同。

## 四、實測（Claude Code）

2026-11-15，Claude Code 2.1.325，MCP Python SDK 2.3.0，在空資料夾 `~/mcp-test` 以 `PART_ASSISTANT_USER=alice` 啟動（alice 的權限是 internal）。檢索在 RTX 5070 Laptop 上執行。下表是 server 端的稽核紀錄。server 看不到 Claude 的回答，回答內容以畫面為準。

| 問題 | server 收到的工具呼叫 | 結果 |
|---|---|---|
| PSU-3200-54V 還有多少庫存？ | `lookup_part` → `check_stock` | ✅ |
| PSU-3200-54V 滿載 hold-up time？ | `search_docs("TEC3200-54-074NA hold-up time full load")` | ✅ 照 server 說明先查料號，再用原廠型號搜尋 |
| TEC3200 進料檢驗要燒機多久？ | `search_docs`：前 3 筆都是 `wi-iqc-0032-revB.md` | ✅ 只用現行版，沒有 rev A |
| TEC3200 的採購單價？ | `search_docs`：5 筆全是公開 datasheet | ✅ internal 看不到 confidential |
| 產線冒煙，開高嚴重度工單 | `create_ticket` → **表單跳出** → TK-0003，`created_by = alice`、`approved_by = Alice` | ✅ elicitation 在真的 host 上可以用 |
| 「另一起通報」再開一張高嚴重度單（同料號） | `create_ticket` → 沒跳表單，回傳 TK-0003、`duplicate: true` | ⚠️ 設計如此，見下 |

這次也確認了：
- `PART_ASSISTANT_USER` 會從 Claude Code 傳到它啟動的 server：稽核紀錄的使用者是 alice。
- 拒絕表單的路徑這次沒有實測（兩次都沒按拒絕），由自動測試涵蓋。

**最後一題的決定**：同一次對話裡，同一個料號、同一個嚴重度只會有一張單，即使使用者真的想再開一張。另一種做法是把標題和說明的 hash 加進 key：內容完全相同的重試照樣會被擋，但模型換個說法重送時，就會多開一張單（低、中嚴重度沒有人工把關）。重複開單在工廠的代價（兩組人處理同一件事），比「另一起事件請開新對話」大，所以**維持現狀**。

## 五、限制

- **身分是自稱的**：誰能啟動 server，誰就能設定 `PART_ASSISTANT_USER`。正式系統要用 HTTP 傳輸加 OAuth，由 server 驗證 token。
- **自己開單、自己核准**：實測時 alice 開的單是 alice 自己核准的。表單裡的名字是自由輸入，也沒有檢查「核准人 ≠ 建立者」（職責分離）。
- **只支援 stdio**：一個 host 對話一個 server 程序，只能本機使用。多人共用要改成 Streamable HTTP，idempotency key 也要改成依 MCP session 產生，不能再依程序。
- **看不到問題和回答**：稽核紀錄只有工具呼叫。問了什麼、答了什麼，要看 host 的紀錄。
- **host 能讀檔案就能繞過**：權限過濾只保護「經過工具」的資料。所以 host 和資料要分開放（見上面「為什麼建議在空資料夾用」）。
- **新增文件要重啟 server**：段落在第一次搜尋時載入，之後就留在記憶體裡。版本的生效日則每次搜尋都重新計算，不用重啟。

## 六、常見問題（Q&A）

### Q：為什麼 server 不用 API key？
server 只執行工具：查 SQLite、做檢索。決定要呼叫哪個工具、把結果寫成回答，是 host 的模型在做。bge-m3 也是模型，但它是本機的檢索模型，不產生文字，也不花錢。

### Q：為什麼核准要用 elicitation，不在工具上加一個 `approved_by` 參數？
參數是模型填的。模型可以自己填一個名字，也可能被 prompt injection 誘導去填。elicitation 是 server 直接請 **host 問人**，答案不經過模型。

### Q：host 不支援 elicitation 怎麼辦？
高嚴重度工單一律失敗、不寫入。寧可開不了單，也不要沒人確認就寫入。低和中嚴重度不受影響。

### Q：為什麼不同的 host 都能用？
MCP 是公開協定：server 宣告有哪些工具、參數的 JSON schema，host 照協定呼叫。同一個 server 不用改程式，就能接到 Claude Code、Claude Desktop、MCP Inspector，或其他支援 MCP 的程式。

### Q：Claude Code 已經能讀檔案、跑 SQL，為什麼還需要 MCP server？
因為規則。直接讓模型跑 SQL，它可以自己在 `approved_by` 填一個名字，`INSERT` 一張高嚴重度工單：資料庫的 CHECK 只檢查有沒有名字，不知道是不是真的有人核准。它也可以讀到不該看的文件。工具把能做的事限縮成 4 個，每個都帶著自己的規則：冪等、人工確認、權限、稽核。
