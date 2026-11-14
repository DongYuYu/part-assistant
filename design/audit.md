# 稽核紀錄：誰問了什麼、呼叫了哪些工具、改了什麼資料

一句話：**每個問題（MCP 則是每次工具呼叫）都寫一筆到資料庫的 `audit_log`，資料庫用 trigger 拒絕修改和刪除；工單本身也記下是誰要求開的。**

程式：`src/part_assistant/audit.py`、`db.py`（資料表和 trigger）、`agent.py`（`_audit`、`approve`）、`mcp_server.py`（`audited`）。
測試：`tests/test_audit.py`。

## 目錄

- [一、為什麼 trace 不夠](#一為什麼-trace-不夠)
- [二、記了什麼](#二記了什麼)
- [三、使用者是誰](#三使用者是誰)
- [四、怎麼查](#四怎麼查)
- [五、限制](#五限制)
- [六、常見問題（Q&A）](#六常見問題qa)

## 一、為什麼 trace 不夠

`traces/*.json` 是除錯用的：每次模型和工具呼叫的耗時、token、成本。它不適合當稽核紀錄：

| | trace | 稽核紀錄 |
|---|---|---|
| 給誰看 | 開發者（為什麼慢、為什麼貴） | 主管、品保、資安（誰做了什麼） |
| 存在哪 | 一題一個 JSON 檔 | 和工單同一個資料庫的 `audit_log` 資料表 |
| 能不能改 | 任何人都能刪、能改 | 資料庫拒絕 `UPDATE` 和 `DELETE` |
| 好不好查 | 要自己翻檔案 | `--user alice --action approve` |

trace 檔現在也附上了 `user_id` 和 `session_id`，兩邊可以對得起來。

## 二、記了什麼

### 每個問題一筆 `ask`（CLI）

| 欄位 | 內容 |
|---|---|
| `at`、`user_id`、`session_id`、`channel` | 時間（UTC+8）、誰、哪一次對話、`cli` / `mcp` / `eval` |
| `detail.question`、`detail.answer` | 問題和回答全文 |
| `detail.tools` | 每次工具呼叫：名稱、參數、**結果摘要** |
| `detail.stop`、`degraded`、`cost_usd` | 怎麼結束、是不是降級回答、花了多少錢 |
| `detail.error` | 中途出錯時的錯誤（例如 401） |

結果摘要只記稽核需要的部分，不複製整頁 datasheet：

| 工具 | 摘要 |
|---|---|
| `search_docs` | 來源清單，例如 `["bel-tec3200-54.pdf p.2"]` |
| `check_stock` | 可用總數 |
| `lookup_part` | 料號、廠商型號 |
| `create_ticket` | 工單編號、狀態、嚴重度、是否重複、建立者、核准人 |

### 核准一筆 `approve`

高嚴重度工單由人核准時，另外記一筆：誰核准（`approver`）、誰要求的（`requested_by`）、工單編號。要求的人和核准的人可以不同，兩個都留下。

### MCP 每次工具呼叫一筆 `tool`

用 MCP 時，問題是 host 的模型（例如 Claude Code）在處理，我們的 server 看不到使用者問了什麼，只看得到工具呼叫。所以每次工具呼叫都記一筆。使用者拒絕核准的工單也會記（`status: not_created`，附原因）。

### 工單本身也記建立者

`tickets.created_by` 和工單在**同一個交易**裡寫入：不可能有工單卻不知道是誰開的。稽核紀錄是另外一筆寫入，萬一程式在兩次寫入之間當掉，工單上的 `created_by` 還在。

### 失敗的問題也記

`run()` 用 `try/finally` 寫稽核紀錄。Claude 在第 2 步回 401 時，第 1 步如果已經開了低嚴重度工單，這筆寫入不能因為後面出錯就沒有紀錄。

## 三、使用者是誰

這個專案**沒有登入系統**，使用者 ID 是呼叫的人自己說的，依序取：

1. `--user NAME`（CLI）或 `build_server(user_id=...)`
2. 環境變數 `PART_ASSISTANT_USER`
3. 作業系統的帳號名稱

Docker 容器裡的帳號一律是 `app`，所以 `compose.yaml` 會把主機的 `$USER` 傳進去。

## 四、怎麼查

```bash
uv run python -m part_assistant.audit                          # 最近 20 筆
uv run python -m part_assistant.audit --user alice             # 某個人
uv run python -m part_assistant.audit --action approve         # 所有核准
uv run python -m part_assistant.audit --json --limit 5         # 完整內容
```

實際輸出（降級模式問一題，不花錢）：

```
#1     2026-11-15T00:20:19+08:00  alice        cli  ask     'FAN-4028-HS 還有多少庫存？' tools=check_stock stop=done
```

## 五、限制

- **身分是自稱的**：任何人都能打 `--user 別人的名字`。正式系統要從登入（SSO）取得身分，不能讓使用者自己填。核准人 `approved_by` 也是同樣的問題。
- **能拿到資料庫檔案的人還是能改**：trigger 擋得住程式的 bug 和隨手寫的 SQL，擋不住直接 `DROP TRIGGER` 或刪掉整個檔案。要防這種情況，要把紀錄送到別的系統（例如只能附加的雲端儲存、SIEM），或每筆帶上前一筆的雜湊（hash chain），讓竄改看得出來。`id` 用 `AUTOINCREMENT` 不重複使用，刪掉的編號會留下缺口，算是很弱的偵測。
- **存了問題和回答全文**：如果使用者在問題裡打了個資，也會被存下來。正式上線要訂保存期限和存取權限；但「只能附加」和「到期刪除」互相衝突，要用歸檔或分區的方式處理。
- **用 `ALTER TABLE` 升級的舊資料庫**：`created_by` 欄位補得上，但舊工單的建立者是空的（本來就不知道）。`audit_log` 是新建的資料表，CHECK 和 trigger 都有。
- **評測也會寫稽核紀錄**：寫在評測用的暫存資料庫裡（`channel = eval`），不會混進真的資料庫。

## 六、常見問題（Q&A）

### Q：為什麼稽核紀錄放在 SQLite，不寫成 log 檔？
要能依使用者、動作查詢，也要能用 trigger 擋掉修改；而且和工單在同一個資料庫，備份時一起備份。量大了以後再送到專門的 log 系統。

### Q：為什麼用 trigger 擋 UPDATE 和 DELETE？
和工單的 CHECK 約束同樣的想法：重要的規則放在資料庫，不靠每一段程式記得遵守。以後有人寫了一支「清理舊資料」的腳本，會直接失敗，而不是默默刪掉稽核紀錄。

### Q：稽核寫入失敗時，回答還要給使用者嗎？
現在的做法是丟出錯誤（不吞掉）。稽核紀錄是必要的，寫不進去代表資料庫有問題，應該讓人知道。代價是使用者這題拿不到回答。

### Q：為什麼 MCP 不記問題？
MCP 的 server 只收到工具呼叫，問題在 host 那邊（Claude Code、Claude Desktop），server 看不到。這也是 MCP 的信任邊界：問了什麼要看 host 的紀錄。

### Q：為什麼工單要有 `created_by`，稽核紀錄裡不是已經有了？
兩者寫入的時間點不同。`created_by` 和工單在同一個交易裡，保證一致；稽核紀錄在整題結束後才寫。查工單時也不用再去翻稽核紀錄。
