# 地端模型：vLLM 和量化比較

一句話：**在 8 GB 的筆電 GPU 上，比較 Qwen3 的 4B 和 1.7B、原始和量化版本的記憶體、速度和黃金樣本分數。結論：降級用的地端模型選 4B INT4（AWQ）。在地端 prompt 加一條「照抄搜尋結果的 source 當出處」之後，它從 20/48 進步到 33/48，仍遠低於 Claude 的 48/48，所以平常一律用 Claude。**

程式：`src/part_assistant/local_agent.py`（OpenAI 格式，Ollama 和 vLLM 共用）、`src/part_assistant/bench_llm.py`（速度）、`src/part_assistant/eval_agent.py --local`（品質）。
測試：`tests/test_fallback.py`、`tests/test_bench_llm.py`。

## 目錄

- [一、為什麼要比](#一為什麼要比)
- [二、環境和做法](#二環境和做法)
- [三、記憶體](#三記憶體)
- [四、速度](#四速度)
- [五、品質（黃金樣本）](#五品質黃金樣本)
- [五之一、改地端 prompt 之後](#五之一改地端-prompt-之後)
- [六、結論](#六結論)
- [七、量測時踩到的坑](#七量測時踩到的坑)
- [八、限制](#八限制)
- [九、常見問題（Q&A）](#九常見問題qa)

## 一、為什麼要比

地端模型放在自己的 GPU 上，資料不出公司，也不按 token 計費；但 GPU 記憶體有限。**量化**是把每個參數存得更省空間：

| 格式 | 每個參數 | 4B 模型的權重 |
|---|---|---|
| BF16（原始） | 2 bytes | 約 7.6 GB |
| FP8 | 1 byte | 約 4.2 GB |
| INT4（AWQ） | 0.5 byte | 約 2.5 GB |

省下的記憶體可以放更多 KV cache（每個對話的短期記憶），同時服務更多人；代價是可能答得比較差。要比的就是這三方的取捨：**記憶體、速度、品質**。

## 二、環境和做法

| 項目 | 內容 |
|---|---|
| 硬體 | RTX 5070 Laptop（8 GB，Blackwell，compute capability 12.0），WSL2；啟動時 GPU 可用 6.83 / 7.96 GiB（約 1.1 GiB 被 Windows 佔用） |
| 推論伺服器 | vLLM 0.32.0 官方 Docker 映像 `vllm/vllm-openai:v0.32.0`（vLLM 需要 torch 2.13，專案是 2.11，所以不裝進專案環境）；對照組 Ollama 0.34.2 |
| 模型 | Qwen 官方在 Hugging Face 發布的 Qwen3 系列（阿里巴巴，Apache-2.0）：`Qwen3-4B`、`Qwen3-4B-FP8`、`Qwen3-4B-AWQ`、`Qwen3-1.7B`、`Qwen3-1.7B-FP8`；Ollama 用 `qwen3:4b`（GGUF Q4_K_M） |
| 日期 | 2026-11-15 ～ 2026-11-16 |

vLLM 啟動指令（每個模型一樣，只換模型名稱）：

```bash
docker run --gpus all -p 127.0.0.1:8000:8000 --ipc=host \
  -v vllm-hf:/root/.cache/huggingface vllm/vllm-openai:v0.32.0 Qwen/Qwen3-4B-AWQ \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
```

- `--tool-call-parser hermes`：把 Qwen3 的工具呼叫轉成 OpenAI 格式的 `tool_calls`。
- `--reasoning-parser qwen3`：思考內容放在獨立欄位，不混進回答。
- 模型權重放在 Docker volume `vllm-hf`，不掛使用者的 `~/.cache`（容器用 root 執行，下載的檔案會變成 root 所有）。

量測：

```bash
# 速度：短 prompt（約 60 token）和長 prompt（system prompt + 5 段 datasheet，約 2,400 token），1 人和 4 人同時
PART_ASSISTANT_LOCAL_URL=http://127.0.0.1:8000/v1 uv run python -m part_assistant.bench_llm --model Qwen/Qwen3-4B-AWQ
# 品質：黃金樣本 50 題（2 題已知問題不計分），檢索改用 CPU，GPU 全部留給 vLLM
PART_ASSISTANT_RETRIEVAL_DEVICE=cpu PART_ASSISTANT_LOCAL_URL=http://127.0.0.1:8000/v1 \
  PART_ASSISTANT_LOCAL_MODEL=Qwen/Qwen3-4B-AWQ uv run python -m part_assistant.eval_agent --local
```

兩種都在本機跑，**不花錢**。

## 三、記憶體

數字取自 vLLM 的啟動 log（`--gpu-memory-utilization 0.85`）：

| 模型 | 權重 | KV cache | 可放 token 數 | 結果 |
|---|---|---|---|---|
| 4B BF16 | 7.6 GB（硬碟上） | — | — | **放不進**：權重就超過可用記憶體，沒有空間放 KV cache |
| 4B FP8 | 4.21 GiB | 1.22 GiB | 8,864 | **最長只能 8k**：16k 的請求需要 2.25 GiB，vLLM 拒絕啟動 |
| **4B INT4（AWQ）** | **2.5 GiB** | 2.87 GiB | 20,848 | 可以 |
| 1.7B BF16 | 3.22 GiB | 2.38 GiB | 22,304 | 可以 |
| 1.7B FP8 | 1.91 GiB | 3.73 GiB | 34,928 | 可以 |

**檢索模型也要 GPU**：bge-m3 加 reranker（fp16）跑一次搜尋，GPU 用到約 4.3 GB。8 GB 的卡扣掉 Windows 的 1.1 GB，只剩約 2.5 GB 給 vLLM，連 4B INT4 都放不下。所以：

- 品質評測時檢索改用 CPU（`PART_ASSISTANT_RETRIEVAL_DEVICE=cpu`）。CPU 和 GPU 的檢索結果完全一樣（32 題的 r@5 = 1.00、MRR = 0.91，每題名次都相同），只是每次搜尋從約 0.6 秒變成約 23 秒。
- 正式部署時，檢索模型和 LLM 要放在不同的 GPU，或檢索放在另一台機器。**GPU 規劃要把 embedding 和 reranker 一起算進去。**

## 四、速度

每個請求固定寫 256 token（vLLM 用 `ignore_eos`），思考關閉，同一設定跑 3 輪取中位數。每個請求的開頭都不同，避免伺服器的 prefix cache 讓 TTFT 看起來太快（見[七](#七量測時踩到的坑)）。

| 模型 | 單人 tokens/s（短） | TTFT（長 prompt） | 4 人同時：總 tokens/s（短） | 4 人同時：總 tokens/s（長） | 4 人同時：TTFT（長） |
|---|---|---|---|---|---|
| 4B FP8（8k） | 71 | 0.26 秒 | 276 | 108 | 0.72 秒 |
| **4B INT4（AWQ）** | **115** | 0.53 秒 | **436** | **192** | 1.56 秒 |
| 1.7B BF16 | 98 | 0.22 秒 | 402 | 233 | 0.67 秒 |
| 1.7B FP8 | 160 | 0.11 秒 | 609 | 342 | 0.32 秒 |
| Ollama 4B Q4_K_M | 81 | 0.82 秒 | 90 | 65 | **6.73 秒** |

### 看出來的事

**1. 寫字的速度由「每個 token 要讀多少權重」決定。** 每寫一個 token，都要把全部權重從 GPU 記憶體讀一次：

| 模型 | 權重（GiB）× tokens/s |
|---|---|
| 1.7B FP8 | 1.91 × 160 ≈ 305 |
| 4B AWQ | 2.5 × 115 ≈ 288 |
| 1.7B BF16 | 3.22 × 98 ≈ 316 |
| 4B FP8 | 4.21 × 71 ≈ 299 |

乘積都約 300 GB/s，也就是這張卡實際的記憶體頻寬。decode 受記憶體頻寬限制（memory-bound），所以 **4B FP8 比 4B INT4 慢**：FP8 每個參數 1 byte，INT4 只有 0.5 byte。

**2. 多人同時使用才看得出推論伺服器的差別。** vLLM 把同時進來的請求合併成一批（continuous batching），4 人同時時，總輸出是單人的約 3.8 倍。Ollama 一次只處理一個請求，總量不變，排在後面的人要等：長 prompt 的 TTFT 從 0.8 秒變成 6.7 秒。

**3. 長 prompt 讓總量掉一半以上。** 4 個 2,400 token 的請求要先讀完 prompt（prefill），KV cache 也被佔滿。真實的降級請求都是長的（要帶搜尋結果），所以規劃容量要用長 prompt 的數字。

## 五、品質（黃金樣本）

48 題（50 題扣掉 2 題已知的解析問題）。同一個 system prompt、同樣的工具（地端只有唯讀工具，所以 3 題工單題一定失敗）、同一個批改器，`temperature 0`，開思考。

| 模型 | 伺服器 | 分數 | 照抄引用範本 `(file p.N)` | 數字錯或沒找到 | 沒呼叫工具 |
|---|---|---|---|---|---|
| Claude Opus 5.5 | 雲端 | **48/48（100%）** | 0 | 0 | — |
| qwen3:4b Q4_K_M | Ollama（舊的原生 API，2026-11-14） | 29/48（60%） | — | — | — |
| qwen3:4b Q4_K_M | Ollama（OpenAI 格式） | **26/48（54%）** | 2 | 17 | 8 |
| Qwen3-4B FP8 | vLLM | 23/48（48%） | 10 | 12 | 6 |
| Qwen3-4B INT4（AWQ） | vLLM | 20/48（42%） | 15 | 13 | 4 |
| Qwen3-1.7B BF16 | vLLM | 9/48（19%） | 14 | 31 | 7 |
| Qwen3-1.7B FP8 | vLLM | 9/48（19%） | 12 | 34 | 12 |

（一題約 2%，每個設定只跑一次；差 1 到 3 題屬於誤差範圍。一題可能同時有多種錯誤。）

### 看出來的事

**1. 模型大小影響遠大於量化。** 同樣是 1.7B，BF16 和 FP8 都是 9 題；同樣是 4B，FP8 和 INT4 差 3 題（在誤差範圍附近）。但 4B 和 1.7B 差一倍以上。所以**記憶體有限時，大一點的模型壓得更小，比小模型不壓縮好**：4B INT4 只佔 2.5 GiB，比 1.7B BF16 的 3.22 GiB 還小，分數是兩倍。

**2. 4B 大多錯在「格式」，1.7B 錯在「內容」。** 4B 的數字常常是對的，但把 prompt 裡的範例 `(file p.N)` 照抄，沒填真正的檔名，例如「效率為 96% (file p.2)」。1.7B 是數字本身答錯，或沒查資料就回答。

**3. vLLM 的 4B 比 Ollama 的 4B 少 3 到 6 題，主要差在照抄範本（10–15 次 vs 2 次）。** 原因還沒確認：可能是量化方法不同（AWQ vs GGUF Q4_K_M），也可能是兩邊的 chat template 不同。

**4. 換成 OpenAI 格式後，Ollama 從 29 變成 26。** 差 3 題，在誤差範圍附近，不能斷定是格式造成的。

## 五之一、改地端 prompt 之後

上面的錯誤裡，最多的是照抄 `(file p.N)`。共用的 system prompt 第 1 條寫「Cite datasheet facts as (file p.N)」，`file` 是佔位符，Claude 會填入檔名，小模型卻照字抄。每筆搜尋結果本來就有現成的出處欄位，例如 `"source": "bel-tec3200-54.pdf p.2"`，所以只在**地端專用**的 `SYSTEM_LOCAL` 加了第 9 條：

> To cite, copy the "source" field of the search result you used, exactly as written, e.g. (abc-1234.pdf p.3). Never write the word "file" instead of the real file name.

- 範例檔名是虛構的，不是黃金樣本裡的檔案，避免把答案寫進 prompt。
- Claude 的 prompt 沒改，所以 Claude 的 48/48 不用重跑（也不用花錢）。

| 模型 | 改之前 | 改之後 | 照抄範本 | 沒附正確出處 | 數字錯或沒找到 | 沒呼叫工具 |
|---|---|---|---|---|---|---|
| vLLM 4B INT4（AWQ） | 20/48（42%） | **33/48（69%）** | 15 → **0** | 19 → 10 | 13 → 12 | 4 → 7 |
| Ollama qwen3:4b Q4_K_M | 26/48（54%） | **30/48（63%）** | 2 → **0** | 16 → 10 | 17 → 20 | 8 → 5 |

（2026-11-16 ～ 17，同樣的環境和指令。地端只有唯讀工具，3 題工單題一定失敗，所以地端最多 45/48。）

- **照抄範本的錯誤完全消失**，兩個模型都是 0 次。
- vLLM 4B INT4 進步最多（+13 題）：它原本的數字常常是對的，只是出處寫錯。表格題從 1/6 變成 6/6，相似型號題從 0/4 變成 3/4。
- Ollama 只多 4 題：它本來就很少照抄，主要的錯是數字答錯（17 → 20，沒有改善）。
- **改完之後，vLLM 4B INT4 反而比 Ollama 4B 高 3 題**（誤差範圍附近）。之前「vLLM 比較差」的差距，主要就是出處格式。
- 4B FP8 和 1.7B 沒有用新 prompt 重跑，上一節是它們舊 prompt 的分數。

**教訓**：寫給強模型的 prompt，換到小模型上要重新評測。佔位符、隱含的慣例，強模型會自動理解，小模型會照字面做。

## 六、結論

1. **地端模型只當備援，不取代 Claude。** 最好的地端設定（4B INT4，加上引用規則）答對 33/48，Claude 是 48/48，和 [routing.md](routing.md) 的結論一致。
2. **8 GB 的卡上，降級模型選 4B INT4（AWQ）on vLLM。**
   - 和 4B FP8 比：分數差 3 題（誤差範圍），但快 60%、可放的 token 多 2.4 倍，請求可以到 16k。
   - 和 1.7B 比：分數是兩倍，記憶體還比較小。
   - 和 Ollama 比：加上引用規則後分數多 3 題（33 vs 30，誤差範圍附近），多人同時使用時總量是 3 到 5 倍。只有一個人用時，Ollama 也夠用。
3. **檢索模型和 LLM 不能共用這張 8 GB 的卡**，正式部署要分開。

## 七、量測時踩到的坑

- **prefix cache 讓 TTFT 太好看**：同一個 prompt 重送，伺服器會重用算好的結果。第一次量 Ollama 長 prompt 的 TTFT 只有 0.03 秒，實際是 0.8 秒。所以 `bench_llm` 讓每個請求的開頭都不同。
- **模型提早結束，總量不能比**：第一輪的長 prompt 只寫了 31 到 54 token 就停了，各模型長度不同。改成 vLLM 固定寫滿 256 token（`ignore_eos`）後重量。Ollama 沒有這個開關，但它這次剛好都寫滿 256。
- **Ollama 的 OpenAI 入口不能設定 context 長度**：預設 4096 token，裝不下搜尋結果。要在 Ollama 啟動時設定 `OLLAMA_CONTEXT_LENGTH=16384`（`compose.yaml` 已經設好）。
- **vLLM 啟動卡住一次**：第一輪的 1.7B 在初始化引擎時停了 66 分鐘，第二輪 3 分鐘就好了。原因沒查到，可能是前一個失敗的容器還沒完全釋放 GPU。

## 八、限制

- **每個設定只跑一次**，48 題一題約 2%；`temperature 0` 也不保證每次結果完全相同。
- **只比了 Qwen3 一個家族。** Qwen 是中國公司開發的模型；模型下載到本機執行，資料不會傳回開發者，但企業能不能用，要看公司的資安和法務規定。換成其他模型（Llama、Gemma、Phi，或台灣的 TAIDE、Breeze）不用改程式，只要換 `PART_ASSISTANT_LOCAL_MODEL`，再用同樣的指令重量。
- **沒有比 GPTQ 和 KV cache 量化**（KV cache 用 FP8 可以多放一倍 token）。
- **速度是單機、短時間的量測**，不是長時間壓測；壓測和每千次請求的成本放在閘道的部分。
- **品質評測時檢索在 CPU 上**，所以評測的「總時間」不代表正式使用時的速度。

## 九、常見問題（Q&A）

### Q：為什麼不直接用最快的 1.7B FP8？
它每秒 160 token，但只答對 9 題。降級時答錯的代價（產線照著錯的規格做）遠大於慢一點。速度要在品質達標之後才比。

### Q：為什麼 FP8 比 INT4 大，分數卻沒有明顯比較好？
FP8 比較接近原始模型，理論上應該比較準；這次多 3 題，在誤差範圍內。要確定，需要更多題目或每個設定多跑幾次。就算 FP8 真的好一點，它在這張卡上只能放 8k token、服務約 1 個人，實用性比較差。

### Q：為什麼 vLLM 比 Ollama 快那麼多？
單人時差不多（115 vs 81 tokens/s，量化格式也不同）。差別在多人同時：vLLM 用 continuous batching 和 PagedAttention，把多個請求放在同一批計算；Ollama 預設一次處理一個。

### Q：照抄 `(file p.N)` 可以修嗎？
已經修了，見[五之一](#五之一改地端-prompt-之後)：只在地端專用的 `SYSTEM_LOCAL` 要求照抄搜尋結果的 `source` 欄位，照抄範本從 15 次變成 0 次，4B INT4 從 20/48 變成 33/48。
