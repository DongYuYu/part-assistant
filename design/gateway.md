# 推論閘道（gateway）

一句話：**app 的所有模型呼叫都先經過閘道。閘道確認是誰在用、有沒有超過額度，再轉給對應的模型，並記下用了多少 token、花了多少錢。**
閘道不決定要用哪個模型（那是 agent 的事，見 [routing.md](routing.md)），也不轉換格式。

程式：`src/part_assistant/gateway.py`。測試：`tests/test_gateway.py`（假的上游，不連網路、不花錢）。

## 目錄

- [一、為什麼要有閘道](#一為什麼要有閘道)
- [二、怎麼運作](#二怎麼運作)
- [三、怎麼用](#三怎麼用)
- [四、設計取捨](#四設計取捨)
- [五、實測](#五實測)
- [六、限制和下一步](#六限制和下一步)
- [七、常見問題（Q&A）](#七常見問題qa)

## 一、為什麼要有閘道

沒有閘道時，每個 app 都自己拿著真的 API key 直接呼叫 Claude：

| 問題 | 沒有閘道 | 有閘道 |
|---|---|---|
| 金鑰 | 每個 app、每台電腦都有真的 key；外洩要全部換 | 真的 key 只在閘道；每個人有自己的閘道金鑰，可以單獨停用 |
| 誰用了多少 | 只看得到整個帳號的帳單 | 每次呼叫一筆：誰、哪個模型、多少 token、多少錢 |
| 失控 | 一個寫壞的迴圈可以一晚燒掉整個月的預算 | 每人每分鐘的請求數、每天的 token 數有上限 |
| 能用的模型 | app 想用哪個就用哪個 | 只放行有核准、知道價格的模型 |
| 雲端和地端 | 兩套做法 | 同一套管理：地端模型也記用量，之後可以算 GPU 成本 |

這些規則和模型、和 app 無關。放在閘道，所有 app 共用一份；放在 app 裡，每個 app 都要再寫一次。

## 二、怎麼運作

```mermaid
flowchart LR
    A[agent<br/>Anthropic SDK] -->|/v1/messages<br/>閘道金鑰| G
    L[LocalAgent<br/>降級時] -->|/v1/chat/completions<br/>閘道金鑰| G
    subgraph G[閘道]
        K[1. 金鑰 → 使用者] --> R[2. 每分鐘請求數] --> Q[3. 今天的 token 數] --> M[4. 模型有沒有核准]
    end
    G -->|真的 API key| C[Claude API]
    G --> V[vLLM / Ollama]
    G --> U[(gateway_usage<br/>每次呼叫一筆)]
```

| 路徑 | 格式 | 轉給 |
|---|---|---|
| `POST /v1/messages` | Anthropic | Claude，原封不動（逐位元組轉發，查詢字串 `?beta=true` 也帶過去） |
| `POST /v1/chat/completions` | OpenAI | 地端的 vLLM 或 Ollama |
| `GET /healthz` | | 健康檢查 |

檢查的順序是「便宜的先做」：

1. **金鑰**：從 `x-api-key`（Anthropic SDK）或 `Authorization: Bearer`（OpenAI 格式）取得，算 sha256 後查表。不對就回 401，不會呼叫上游。
2. **每分鐘請求數**：60 秒滑動視窗，每人各自計算。有設定 `PART_ASSISTANT_REDIS_URL` 時，計數放在 Redis，所有副本共用；沒有設定時，放在程序自己的記憶體（見下面的「速率限制和 Redis」）。超過就回 429，加上 `retry-after: 60`。
3. **今天的 token 數**：從用量表加總今天（UTC）的 token。用完就回 429，加上 `x-should-retry: false`（Anthropic SDK 看到這個 header 就不會自動重試，因為幾秒後重試也沒用）。
4. **模型**：Claude 的模型必須在價格表（`agent.PRICES`）裡，否則回 400。沒有價格的模型就算不出成本，所以不放行。

錯誤一律用 app 預期的格式回傳（Anthropic 或 OpenAI），讓 SDK 丟出平常的例外。上游連不上時回 502，逾時回 504，agent 會照原本的規則降級（見 [routing.md](routing.md)）。

**用量**：每次呼叫寫一筆到 `gateway_usage`。

- 不串流：從回應的 `usage` 讀取。
- 串流：資料原封不動地轉給 app，同時從 SSE 事件讀出 token 數。
  - Claude：輸入在 `message_start`，輸出在 `message_delta`。
  - OpenAI 格式：閘道會加上 `stream_options.include_usage`，伺服器就會在最後一個 chunk 附上用量。
- app 中途斷線的話，已經產生的部分照樣記帳。
- Claude 的成本照價格表算，包含快取讀寫。地端記成 0（GPU 是固定成本，見下面的 Q&A）。

### 速率限制和 Redis

只有一個閘道時，計數器放在記憶體就夠了。但跑 3 個副本時，每個副本各算各的，每分鐘 30 次的使用者實際上能打 90 次。把計數器搬到 Redis，所有副本共用同一份。

**做法**：每個使用者一個 sorted set（`gateway:rpm:<user>`），每個請求是一筆，分數是時間（毫秒）。一個 Lua 腳本做完四件事：

1. 刪掉 60 秒前的紀錄（`ZREMRANGEBYSCORE`）。
2. 數剩下幾筆（`ZCARD`）。
3. 已經滿了就拒絕；還沒滿就加入這一筆（`ZADD`）。
4. 設定 60 秒後過期（`PEXPIRE`）：使用者不再來，key 會自己消失。

- **為什麼用 Lua 腳本**：Redis 執行腳本時，中間不會插入其他指令。如果分成幾個指令送出，兩個副本可能同時看到 29 筆，都判斷「還沒滿」，結果變成 31 筆（race condition）。
- **時間用 Redis 的 `TIME`**：每台機器的時鐘可能差幾百毫秒，統一用 Redis 的時鐘。
- **連線逾時 0.2 秒**：Redis 變慢時，不能拖慢每一個請求。
- **Redis 不存硬碟**（`--save ""`）：計數器只需要記得 60 秒，Redis 重啟頂多讓計數歸零一分鐘。
- **固定視窗（`INCR` 加 `EXPIRE`）比較簡單**，每人只要一個整數。但它在視窗交界處可能放行兩倍：在 0:59 打 30 次，1:00 又打 30 次。所以這裡選滑動視窗，代價是每人要存最多 rpm 筆紀錄。

**實測**（2026-11-17，本機兩個閘道程序 8101 和 8102，金鑰每分鐘 4 次，輪流送 8 個請求到 Ollama）：

| 設定 | 結果 |
|---|---|
| 不用 Redis | 8 個全部 200：每個副本各放行 4 個 |
| 用 Redis（`redis:8.8.4-alpine`） | 前 4 個 200，後 4 個 429，不管送到哪個副本 |

測試（`tests/test_gateway.py`）：兩個副本共用 Redis、視窗過後恢復、key 會過期、Redis 連不上時放行並計數。CI 有 Redis 服務，所以這些測試會在 CI 執行；本機沒有 Redis 時會自動跳過。

### 監控指標（`GET /metrics`）

給 Prometheus 定期來抓（pull）。依 RED 方法（Rate 請求數、Errors 錯誤、Duration 耗時），再加上 token 和成本：

| 指標 | 類型 | label | 回答什麼問題 |
|---|---|---|---|
| `gateway_requests_total` | Counter | route, model, status | 每秒多少請求？錯誤率多少？ |
| `gateway_request_duration_seconds` | Histogram | route | P95 延遲多少？ |
| `gateway_tokens_total` | Counter | route, model, kind（input / output） | 每分鐘用多少 token？ |
| `gateway_cost_usd_total` | Counter | model | 每小時花多少錢？ |
| `gateway_rejections_total` | Counter | route, reason（auth / rate / quota / model / bad_request） | 閘道擋下多少？為什麼？ |
| `gateway_in_flight_requests` | Gauge | route | 現在有幾個請求在等模型回答？ |

- **沒有 user label**：每一種 label 組合都是一條時間序列，使用者有 1 萬個就是 1 萬倍。每人的用量看 SQLite 的 `gateway_usage`。
- **地端的 model 名稱只有成功時才當 label**：名稱是 app 填的，失敗時一律記成 `-`，否則亂填名稱就能一直製造新序列。
- **耗時的 bucket 是 0.25 到 120 秒**：預設的 bucket 是 5 毫秒到 10 秒，給網頁用的，LLM 呼叫大多會落在最後一格。
- **不需要金鑰**：正式環境只讓監控網段連得到這個 port。

### 用 Prometheus 抓指標

```bash
uv run --env-file .env python -m part_assistant.gateway serve      # 閘道在本機的 8100
docker compose -f compose.monitoring.yaml up -d                     # Prometheus 每 5 秒抓一次
# 瀏覽器開 http://localhost:9090 → Status → Targets：gateway 應該是 UP
```

- 設定檔是 `monitoring/prometheus.yml`。Prometheus 用主機的網路（`network_mode: host`），所以閘道只要聽 127.0.0.1，不用對外開放。
- 映像檔鎖定 `prom/prometheus:v3.15.1`，資料保存 7 天。

常用查詢（PromQL）：

| 想知道 | 查詢 |
|---|---|
| 每秒請求數 | `sum(rate(gateway_requests_total[5m]))` |
| 錯誤率（非 2xx） | `sum(rate(gateway_requests_total{status!~"2.."}[5m])) / sum(rate(gateway_requests_total[5m]))` |
| P95 延遲 | `histogram_quantile(0.95, sum by (le) (rate(gateway_request_duration_seconds_bucket[5m])))` |
| 平均延遲 | `sum(rate(gateway_request_duration_seconds_sum[5m])) / sum(rate(gateway_request_duration_seconds_count[5m]))` |
| 每分鐘 token | `sum by (kind) (rate(gateway_tokens_total[5m])) * 60` |
| 每小時花費 | `sum(rate(gateway_cost_usd_total[1h])) * 3600` |
| 被擋下的原因 | `sum by (reason) (increase(gateway_rejections_total[10m]))` |
| 閘道活著嗎 | `up{job="gateway"}` |

### Grafana 面板

`docker compose -f compose.monitoring.yaml up -d` 會一起啟動 Grafana 13.3.0：http://localhost:3000，不用登入就能看；要修改需要 admin。

- **面板寫成檔案（provisioning）**：資料來源在 `monitoring/grafana/provisioning/`，面板在 `monitoring/grafana/dashboards/gateway.json`。不在網頁上點出來，因為那樣的設定只存在 Grafana 的資料庫，換一台電腦就沒了，也不能 code review。`allowUiUpdates: false`：以 git 裡的檔案為準。
- **排版**：
  - 第一排：4 個數字，回答「現在正常嗎」：閘道狀態、每秒請求數、上游錯誤率、過去 1 小時的 Claude 花費。
  - 第二排：請求數和延遲**左右並排**。兩邊一起升高通常是排隊；只有延遲升高，是上游變慢。
  - 第三、四排：每分鐘的 token、被擋下的原因、進行中的請求、每小時花費。
- **花費的序列也先設為 0**：價格表裡的模型是已知的，所以沒有 Claude 流量時，面板顯示 $0，而不是看起來像壞掉的「No data」。

**實測時踩到的三件事**（2026-11-17，Ollama qwen3:4b 經過閘道）：

1. **第一次的錯誤不見了**：第一版只有在錯誤發生時才建立那條序列。Prometheus 第一次抓到時值已經是 3，沒有之前的值可以相減，`increase()` 就算成 0。改成啟動時先把已知的組合（route × reason）設為 0 之後，同樣送 3 次錯誤金鑰，就能查到 3.22。狀態碼和模型名稱無法事先列舉，所以 `requests_total` 的新序列還是有這個問題。告警可以改用 `requests_total` 加上 `rejections_total` 一起判斷，或者在 Grafana 的面板上看原始值。
2. **`increase()` 不是整數**：3 次顯示成 3.22。因為 Prometheus 會把取樣到的區段等比例外推到整個時間範圍。看趨勢沒問題，但不要拿它來對帳；對帳用 SQLite 的用量表。
3. **重啟不會讓數字變成負的**：閘道重啟後 counter 從 23 掉回 0。`rate()` 和 `increase()` 會把下降判斷為重啟，從 0 接著算。所以 counter 一定要用這兩個函數來看，不要直接拿原始值相減。

## 三、怎麼用

```bash
# 1. 發金鑰（只印一次；資料庫只存 sha256）
uv run python -m part_assistant.gateway add-key alice
uv run python -m part_assistant.gateway add-key bob --rpm 10 --daily-tokens 200000

# 2. 啟動閘道（真的 ANTHROPIC_API_KEY 只有閘道讀得到）
PART_ASSISTANT_UPSTREAM_LOCAL=http://127.0.0.1:8000/v1 \
    uv run --env-file .env python -m part_assistant.gateway serve      # http://127.0.0.1:8100

# 3. app 改走閘道：只改環境變數，程式不用改
export ANTHROPIC_BASE_URL=http://127.0.0.1:8100
export ANTHROPIC_API_KEY=pa-...                     # 閘道金鑰，不是真的 key
export PART_ASSISTANT_LOCAL_URL=http://127.0.0.1:8100/v1
export PART_ASSISTANT_LOCAL_API_KEY=pa-...
uv run python -m part_assistant.agent "PSU-3200-54V 還有多少庫存？"

# 4. 看用量、停用金鑰
uv run python -m part_assistant.gateway usage [--day 2026-11-17]
uv run python -m part_assistant.gateway revoke alice
```

用 Docker 跑（`compose.yaml` 的 `gateway` 服務）：

```bash
docker compose run --rm gateway add-key alice     # 金鑰存在 volume 裡
docker compose up -d gateway                      # 127.0.0.1:8100；用服務名稱連 ollama
docker compose run --rm gateway usage
```

- 映像檔是 `Dockerfile.gateway`，**265 MB**（加 Redis 套件之前是 257 MB）。app 的映像檔是 12.3 GB，大部分是 torch。
- 它只安裝 `pyproject.toml` 裡的 `gateway` 依賴群組（版本照樣由 `uv.lock` 鎖定），只複製 `gateway.py`、`prices.py` 和 `__init__.py`。
- 價格表因此從 `agent.py` 搬到獨立的 `prices.py`。原本閘道為了一張價格表 import `agent.py`，連帶載入了 numpy、rank_bm25、pymupdf 和 anthropic。現在有測試確保閘道不會 import 到這些。
- 以非 root 使用者執行。port 只開在本機的 127.0.0.1。

`PART_ASSISTANT_UPSTREAM_LOCAL` 預設是 Ollama（`http://127.0.0.1:11434/v1`）。用量資料庫預設是 `data/processed/gateway.db`，可以用 `PART_ASSISTANT_GATEWAY_DB` 改。

## 四、設計取捨

| 決定 | 為什麼 | 放棄了什麼 |
|---|---|---|
| **不轉換格式**：Anthropic 格式給 Claude，OpenAI 格式給地端 | 轉換會丟掉 Claude 的功能，例如 strict tool、prompt caching、thinking。Anthropic 自己的 OpenAI 相容層也只建議拿來測試 | 不能只改一個模型名稱就把 Claude 換成地端模型。但降級時本來就要換 prompt 和工具（`SYSTEM_LOCAL`、唯讀工具），單純換模型名稱也不夠 |
| **Claude 的請求逐位元組轉發** | 閘道不需要懂每個新功能，Anthropic 加了新參數也不用改閘道 | 不能在閘道改寫請求內容（之後做 PII 遮罩就要解開來改） |
| **閘道不決定模型** | 降級需要的不只是換模型（還要改 prompt、拿掉開單工具、加警告），這些是業務邏輯，放在 agent | 法規類的規則（例如機密資料只能用地端）之後可以加在閘道 |
| **金鑰只存 sha256** | 資料庫外洩也拿不到金鑰；這和存密碼的做法相同 | 金鑰忘了就只能發新的 |
| **配額是請求前檢查** | 輸出長度事先不知道 | 每天的最後一個請求可能超過，最多超過一個請求的量（多數雲端服務也這樣做） |
| **速率計數器放 Redis**（沒設定時放記憶體） | 多個副本共用同一個計數，才不會各算各的 | 多一個要維運的服務；Redis 掛掉時速率限制暫時失效（放行） |
| **Redis 掛掉時放行（fail open）** | 速率限制是保護措施，不該變成新的故障點；每日配額存在資料庫，仍然有效 | 那段時間可能被多打一些；用 `gateway_rate_limiter_errors_total` 讓人看得到 |
| **用 Starlette，不用 FastAPI** | MCP SDK 已經帶了 Starlette、uvicorn、httpx2，不必新增套件；閘道只有 3 個路徑，不需要自動產生文件 | 沒有自動的 OpenAPI 文件 |

## 五、實測

2026-11-17，RTX 5070 Laptop，Ollama 0.34.2 的 qwen3:4b：

- `curl` 打 `/v1/chat/completions`：正常回答；用錯的金鑰回 401（OpenAI 格式），上游沒有收到請求。
- **LocalAgent 不改程式**，只把 `PART_ASSISTANT_LOCAL_URL` 指向閘道：問「PSU-3200-54V 還有多少庫存？」，呼叫了 `lookup_part`、`check_stock`，回答 296。用量表記了 3 次模型呼叫，共 2,829 個輸入 token 和 1,970 個輸出 token。
- Claude 這條路：測試裡用**真的 Anthropic SDK** 經過閘道（上游是假的），確認 `?beta=true`、`anthropic-version` 有帶過去，真的 key 換上去了，閘道金鑰沒有送出去。
- **真的 agent 經過閘道問 Claude**（agent 程式沒改，只設 `ANTHROPIC_BASE_URL` 和閘道金鑰；真的 key 只有閘道讀得到）：問「PSU-3200-54V 還有多少庫存？」，呼叫 `check_stock`，回答 296，並列出三個倉庫。
  - 閘道記了 2 次 Claude 呼叫，共 3,170 個輸入 token 和 214 個輸出 token，**$0.0130**。
  - agent 自己的 trace 也是 **$0.0130**：兩邊各自計算，結果一致。

閘道本身增加多少延遲還沒量，會在壓測那一步一起量。

## 六、限制和下一步

- **閘道是單點故障**：閘道掛掉時，Claude 和地端兩條路都斷，降級也沒用。正式環境要跑多個副本，放在負載平衡器後面。速率計數器已經可以放 Redis，但金鑰和用量還在 SQLite，多台機器要改成 Postgres。
- **配額用完時 agent 還是會試著降級**：429 對 agent 來說就是「Claude 暫時不能用」，所以會改走地端，而地端也經過同一個閘道，同樣被拒。結果正確（不會多花錢），只是多一次呼叫，錯誤訊息也不夠清楚。之後可以讓 agent 分辨「配額用完」和「Claude 忙」。
- **身分**：閘道金鑰是一人一把，但還不是 SSO。正式系統要由 SSO 換發短效 token。
- 還沒做的（第二階段後面幾步）：
  - 告警規則（例如錯誤率 > 5% 持續 5 分鐘、閘道 down）
  - 輸入端的 PII 遮罩、輸出端的 schema 驗證
  - 用 k6 或 locust 壓測，換算每千次請求的成本（包括地端 GPU 攤提）

## 七、常見問題（Q&A）

> 「為什麼不直接用 LiteLLM 這類現成的閘道？」

正式環境我會先評估現成的產品（LiteLLM、雲端廠商的 API gateway），因為它們已經做好多副本、面板和大量的模型轉接。自己寫一個小的，是為了講得出每一層在做什麼：金鑰雜湊、滑動視窗、配額、串流時怎麼記帳、錯誤格式怎麼讓 SDK 照常降級。評估現成產品時，這些就是要問的問題。

> 「串流的時候怎麼知道用了多少 token？」

資料一邊轉給 app，一邊解析 SSE 事件。Claude 在 `message_start` 給輸入 token，在 `message_delta` 給輸出 token 的累計值。OpenAI 格式要在請求加上 `stream_options.include_usage`，伺服器才會在最後附上用量，所以閘道會自己加上。事件可能被切在任何位置，所以先累積到換行才解析；測試用每 7 個位元組切一段來驗證。

> 「地端模型成本記 0，合理嗎？」

每次呼叫的邊際成本接近 0，但 GPU 是固定成本。用量表記下了地端的 token 數，壓測量出每秒能處理多少 token 之後，就能換算「GPU 每月成本 ÷ 每月 token 數」，得到每千 token 的攤提成本，再和 Claude 比較。

> 「為什麼 429 有時候叫 SDK 不要重試？」

每分鐘的上限過一分鐘就恢復，所以回 `retry-after: 60`，讓 SDK 等一下再試。每天的配額要等到明天才恢復，SDK 在幾秒內重試兩次只是浪費，所以加上 `x-should-retry: false`。這是 Anthropic SDK 支援的 header，測試裡有驗證。

> 「真的 key 放在閘道，閘道被攻破怎麼辦？」

風險集中到一個地方，反而比較好管：只有閘道需要保護和輪替，也只有閘道需要讓資安稽核。各 app 拿到的閘道金鑰可以個別停用，外洩的影響範圍只有那一個人的配額。
