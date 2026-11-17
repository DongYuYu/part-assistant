# HTTP 服務（`POST /ask`）

一句話：**把料號助手從「每問一題就啟動一次的 CLI」改成一直開著的服務。檢索模型只在啟動時載入一次，之後每題都直接使用 GPU 上已經載入好的模型。**

程式：`src/part_assistant/server.py`。測試：`tests/test_server.py`、`tests/test_audit.py`（真的 agent 加上 HTTP 服務的那一題）。

## 目錄

- [一、為什麼要寫成服務](#一為什麼要寫成服務)
- [二、怎麼運作](#二怎麼運作)
- [三、怎麼用](#三怎麼用)
- [四、實測](#四實測)
- [五、第一次實測失敗：稽核表不認得新的管道](#五第一次實測失敗稽核表不認得新的管道)
- [六、限制和下一步](#六限制和下一步)
- [七、常見問題（Q&A）](#七常見問題qa)

## 一、為什麼要寫成服務

RTX 5070 Laptop 上實測（2026-11-17）：

| 階段 | 時間 |
|---|---|
| 載入 bge-m3 和 reranker 到 GPU | 13.6 秒 |
| 第一次搜尋 | 1.17 秒 |
| 之後每次搜尋 | 0.61 秒 |
| GPU 記憶體 | 3.9 GB |

CLI 每問一題就是一個新的程序，每題都要再付一次 13.6 秒的載入時間。服務只在啟動時載入一次。

其他理由：

- **使用者不會用 CLI**：網頁和通訊軟體的機器人，都是透過 HTTP 呼叫後端。
- **多人共用一份模型**：10 個人同時用 CLI，就要 10 份模型（10 × 3.9 GB），8 GB 的卡連兩份都放不下。
- **集中管理**：更新、身分驗證、稽核都只在伺服器上做一次。

## 二、怎麼運作

```
啟動 → 載入檢索模型（約 14 秒）→ 才開始監聽 port → /healthz 有回應就代表可以回答

POST /ask {"question": "..."}，標頭 X-User
  → 在背景執行緒裡跑 agent（和 CLI 同一個 agent：同樣的工具、規則、降級、稽核）
  → 回傳 answer、stop、degraded、cost_usd、用了哪些工具、session_id、待核准的工單草稿
```

| 設計 | 為什麼 |
|---|---|
| 模型載入完才開始監聽 | `/healthz` 有回應就代表真的能回答。K8s 的 readiness 檢查、compose 的 healthcheck 都能直接用 |
| agent 在背景執行緒裡跑 | 呼叫 Claude、查資料庫、在 GPU 上搜尋都會卡住。放在 event loop 上的話，一個人在等，所有人都要跟著等。測試：4 個各需 0.5 秒的問題同時送進來，總共不到 1.5 秒 |
| 檢索器共用，加一把鎖 | GPU 上只放一份模型。不確定它能不能被多個執行緒同時使用，所以一次只搜尋一個。搜尋只要 0.6 秒，等 Claude 回應的那幾秒仍然可以並行 |
| 每個請求開自己的資料庫連線 | Python 的 sqlite3 連線不能跨執行緒使用。開一條連線很便宜 |
| Claude 的 client 共用 | 可以重複使用 HTTP 連線 |
| 斷路器的狀態放記憶體 | CLI 每題都是新的程序，所以要寫進檔案；服務是一直開著的程序，放記憶體就夠了 |
| 一定要帶 `X-User` | 稽核要知道是誰問的。這是 SSO 的暫代做法：只能放在會驗證身分、並且由它設定這個標頭的 proxy 後面 |
| 高嚴重度工單只回傳草稿 | 核准必須由一個身分可信的人來做（另外一個步驟，還沒做） |

## 三、怎麼用

```bash
# 本機
uv run --env-file .env python -m part_assistant.server        # http://127.0.0.1:8000
curl -s 127.0.0.1:8000/ask -H "X-User: alice" -H "Content-Type: application/json" \
    -d '{"question": "PSU-3200-54V 還有多少庫存？"}'

# Docker：所有模型呼叫都經過閘道
docker compose run --rm gateway add-key api        # 印出閘道金鑰
# 把它寫進 .env：PART_ASSISTANT_GATEWAY_KEY=pa-...
docker compose -f compose.yaml -f compose.gpu.yaml --profile api up -d api
```

- Compose 的 `api` 服務**不讀 `.env`**（沒有 `env_file`），只拿到閘道金鑰。真的 `ANTHROPIC_API_KEY` 只有閘道有。
- `ANTHROPIC_BASE_URL` 和 `PART_ASSISTANT_LOCAL_URL` 都指向閘道，所以 Claude 和降級時用的地端模型都經過閘道記帳。
- port 只開在 127.0.0.1，因為 `X-User` 是直接信任的。

## 四、實測

2026-11-17，WSL2，RTX 5070 Laptop，`api` 在 Docker 裡使用 GPU：

| 項目 | 結果 |
|---|---|
| `compose up`（包括每次都會跑的 prepare） | 25 秒 |
| 之後到 `/healthz` 開始回應（載入模型） | 13 秒 |
| 沒帶 `X-User` | 401 |
| GPU 記憶體（api，再加上已經啟動的 Ollama） | 4.3 GB / 8 GB |
| 「PSU-3200-54V 還有多少庫存？」 | 5.0 秒，**$0.0133**，呼叫 `check_stock`，回答 296 個並列出 3 個倉庫的明細 |
| 稽核紀錄 | `dongyu  api  ask  … tools=check_stock stop=done` |
| 閘道用量 | 記在 `api` 這把金鑰底下 |

## 五、第一次實測失敗：稽核表不認得新的管道

第一次實測問了 2 題，**Claude 都回答了，閘道也記了 $0.055，但兩題都回傳 500**。

- **原因**：稽核表的 CHECK 只允許 `cli`、`mcp`、`eval`，服務用的是新的 `api`。稽核紀錄寫不進去，所以不回傳答案。「沒有稽核紀錄就沒有答案」正是我們想要的行為，但管道清單漏了 `api`。
- **為什麼測試沒抓到**：服務的測試用的是假的 agent，假的 agent 不會寫稽核紀錄。
- **修正**：
  1. 管道清單加上 `api`。
  2. SQLite 不能修改既有的 CHECK，所以 `init_schema` 會在同一個 transaction 裡重建舊的稽核表：每一筆紀錄連同 id 一起複製，再重建索引和「只能新增」的 trigger。
  3. 寫升級測試時又抓到一個 bug：重建後，`AUTOINCREMENT` 的計數器會不見，被刪掉的號碼會被重新使用，「有紀錄被刪過」的證據就消失了。現在計數器也一起搬過去，包括表已經被清空的情況。
  4. 新增「真的 agent 加上 HTTP 服務」的測試（只有 Claude 是假的），它會實際寫入稽核紀錄。
- 升級正式的資料庫之前，先備份了 Docker volume 裡的 `factory.db`。

**教訓**：假物件讓測試又快又便宜，但會漏掉元件之間的配合。至少要有一個測試，讓真的元件串在一起跑。

## 六、限制和下一步

- **身分**：`X-User` 只是暫代。正式環境由 SSO 換發 token，伺服器自己驗證。
- **工單核准**：高嚴重度的工單只回傳草稿。核准要另外做一個步驟：草稿存進資料庫，由一個身分可信的核准人確認。
- **一個程序裡同時有 agent 和檢索**：映像檔有 12 GB，而且每個副本都需要一張 GPU。下一步是把檢索拆成獨立的服務：
  - agent 只用 CPU，可以開很多副本。
  - 檢索服務使用 GPU，副本數依 GPU 預算決定。
  - agent 改用 `RemoteRetriever`（和 `Retriever` 有一樣的方法，內部改成發 HTTP 請求），其他程式不用改。
- **還沒放進 Kubernetes**：kind 預設不支援 GPU。要讓 cluster 使用 GPU 容器，可以照 Ollama 的做法，把容器接上 kind 的網路，再用 ExternalName Service 給它一個 cluster 裡的名字（見 [k8s/README.md](../k8s/README.md)）。
- **沒有自己的 `/metrics`**：模型呼叫的指標在閘道上有；agent 自己的指標（每題的步數、工具錯誤、降級次數）還沒有。

## 七、常見問題（Q&A）

> 「為什麼 agent 要放在背景執行緒，不直接寫成 async？」

agent、Anthropic SDK 的同步 client、sqlite3、檢索都是同步的程式。全部改成 async 要動很多地方，對 CLI 和評測也沒有好處。Starlette 的 `run_in_threadpool` 可以讓同步的程式不擋住 event loop，等待 Claude 的時間也能並行。真的需要支援大量同時連線時，再考慮改成 async。

> 「檢索為什麼要加鎖？不會變慢嗎？」

一次搜尋約 0.6 秒，一題的總時間大多花在等 Claude 回應（幾秒）。加鎖只讓「搜尋」這一小段排隊。真正的吞吐量瓶頸在 GPU，拆成獨立的檢索服務之後，可以在那裡做批次處理。

> 「為什麼寫不進稽核紀錄就不回答？」

稽核的目的是「每個答案都查得到是誰、在什麼時候問的」。如果寫失敗還是照樣回答，就會出現查不到紀錄的答案。這次的失敗雖然多花了 $0.055，但沒有任何一個答案沒有留下紀錄。
