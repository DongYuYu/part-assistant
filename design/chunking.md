# 切段（chunk）設計

> 程式：[`src/part_assistant/chunk.py`](../src/part_assistant/chunk.py)、[`parse.py`](../src/part_assistant/parse.py) 的 `parse_pdf_cached`
> 測試：[`tests/test_chunk.py`](../tests/test_chunk.py)
> 執行：`uv run python -m part_assistant.chunk` → `data/processed/chunks.jsonl`

## 目錄

1. [為什麼要切](#一為什麼要切)
2. [輸入是什麼](#二輸入是什麼)
3. [文字的切法](#三文字的切法)
4. [表格的切法](#四表格的切法)
5. [前綴](#五每個-chunk-的前綴)
6. [欄位](#六chunk-的欄位)
7. [chunk_id](#七chunk_id用內容決定-id)
8. [解析快取](#八解析快取)
9. [怎麼驗證](#九怎麼知道切得好不好)
10. [目前的限制](#十目前的限制)
11. [常見問題（Q&A）](#十一常見問題qa)

---

## 一、為什麼要切

RAG 的流程是：問題進來，**找出最相關的幾段文字**，再放進 prompt 讓模型回答。整份 PDF 直接丟給模型行不通，原因有三個：

1. **太長**：TEC1600 有 56 頁。9 份全部放進去，成本高、速度慢，模型也容易漏看。
2. **檢索不準**：一份文件同時講效率、尺寸、保護電路。整份變成一個向量，什麼都講一點，就等於什麼都不準。
3. **出處不精確**：我們要回答「第 2 頁」，而不是「這份 PDF 裡」。

所以要切成**大小適中、主題單一、單獨拿出來也看得懂**的片段，這就是 chunk。切法要在兩種失敗之間取得平衡：

- **太小**：失去上下文。例如只拿到 `| 94 | % |`，不知道是什麼的 94%。
- **太大**：混了好幾個主題，向量變模糊，也浪費 token。

## 二、輸入是什麼

chunk.py 不直接讀 PDF，讀的是 parse.py 輸出的**區塊（Block）清單**，順序和原文相同：

```
Block(kind="text",  page=2, text="## ELECTRICAL SPECIFICATIONS\n\n...")
Block(kind="table", page=2, table=Table(rows=[[...], ...], has_header=True, page=2, page_end=2))
Block(kind="text",  page=2, text="Note:\n\n1. Minimum loading is 5% ...")
```

`chunk_document()` 從頭到尾走一次。**文字和表格走兩條不同的路**，因為它們的結構完全不同。

## 三、文字的切法

### 1. 遇到標題就換章節

以 `#` 開頭的段落是標題，例如 `## **4 PROTECTION**`。遇到標題時：

- 先把手上累積的文字輸出成一個 chunk；
- 再記下新的章節名稱 `section = "4 PROTECTION"`。

標題代表主題切換。把「輸入規格」和「保護電路」混在同一個 chunk，向量會變得四不像。

### 2. 以段落為單位累積，約 400 tokens 輸出一次

文字先用空行切成段落，一段一段放進暫存區 `buf`。如果再加一段會超過 `TEXT_TOKENS = 400`，就先輸出目前的內容。

為什麼以段落為單位，而不是硬切固定字數？因為段落是作者寫好的意義單位。硬切會把句子切成兩半，前後都不完整：

```
... the output voltage shall be within ±5% of |
| nominal under all load conditions ...
```

單一段落本身就超過 400 時（`split_long`），先在句尾（`. ! ? ;`）切開；只有單一句子仍然太長時，才照字數硬切。

### 3. Overlap：下一個 chunk 重複前一個的結尾，約 15%

輸出一個 chunk 後，**保留最後幾個段落（合計不超過 `OVERLAP_TOKENS = 60`）**，當作下一個 chunk 的開頭：

```
chunk A: [段1][段2][段3][段4]
chunk B:             [段4][段5][段6]...   ← 段4 重複
```

有些資訊剛好落在邊界上，例如段 3 寫「在以下條件」，段 4 才列出條件。有了 overlap，至少有一個 chunk 同時包含兩者。代價是多存約 15% 的重複內容。

**overlap 不跨標題**：換章節時呼叫 `flush(keep_overlap=False)` 清空暫存區，避免「輸入規格」的結尾混進「保護電路」的 chunk。

### 4. 清除雜訊

- 拿掉 `**粗體**` 的 `**`。
- 拿掉 `<!-- Start of picture text -->` 這類記號，但保留圖片中的文字（例如 `VIN_GOOD`）。

## 四、表格的切法

規格書的答案大多在表格裡，所以表格的規則最多。

### 1. 表格是原子單位，絕不從一列中間切開

一列就是一個完整的事實：

```
| Hold-up Time | @ 3200 W | 9 | | ms |
```

從中間切開，這個事實就毀了。所以表格一律以**列**為單位處理。

### 2. 大表依列切開，每一段都重複表頭（`table_bodies`）

最大的表格有 198 列，一定要切。做法是一列一列往下加，超過 `TABLE_TOKENS = 600` 就開新的一段，而**每一段的開頭都複製一次表頭**：

```
段 1:                               段 2:
| PARAMETER | MIN | MAX | UNIT |    | PARAMETER | MIN | MAX | UNIT |   ← 表頭重複
| --- | --- | --- | --- |          | --- | --- | --- | --- |
| row 1 ...                         | row 41 ...
```

沒有表頭的話，看到 `| 13.5 | 14.5 | V |` 也不知道哪個是 MIN、哪個是 MAX。

**為什麼表格放寬到 600，文字只有 400？** 表格切開的代價比文字大，所以預算放寬，盡量讓一張表保持完整。

### 3. colspan 去重（`dedup_row`）

parse.py 展開合併儲存格時，會把同一個值複製到每一欄：

```
| Max Input Current | < 16 A at 100 to 240 VAC | < 16 A at 100 to 240 VAC | ...（共 6 次）
```

相鄰而且相同的格子只保留一個：

```
| Max Input Current | < 16 A at 100 to 240 VAC |
```

這樣能省 token，embedding 也不會被重複的字詞拉偏。

**取捨**：去重之後，各列的欄數可能不同，欄位不再完全對齊。對要讀它的 LLM 來說可以接受，但這已經不是格式嚴格的 Markdown 表格。

### 4. 把註腳說明複製進表格（`find_footnotes`）

parse.py 已經把小字上標改寫成 `0 A[^1]`，但說明文字 `1. Minimum loading is 5%…` 位於**表格後面的文字區塊**裡。如果不處理，檢索到這張表時，模型只會看到 `[^1]`，不知道它代表什麼。

處理步驟：

1. 找出這張表用到哪些記號，例如 {1, 2, 3}；
2. 往後看最多 3 個區塊，範圍不超過表格結束頁的下一頁；
3. 找出開頭是 `1.`、`- 2`（清單）、`> 3`（引用）這類格式的行，**而且數字必須是這張表有用到的**；
4. 以 `[^1]: Minimum loading…` 的格式附在表格 chunk 的最後。

實際的結果：

```
[Advanced Energy | CSU2400AT | ae-csu2400at.pdf | p.2 | ELECTRICAL SPECIFICATIONS]
| Max Input Current | < 16 A at 100 to 240 VAC |
| Output Current | 0 A[^1] | - | 196.7 A | 0 A | - | 3 A |
...
[^1]: Minimum loading is 5% of rated current for load response testing.
[^2]: From 5% to 100% of rated current.
[^3]: 2,000µF of external capacitance is required for cold redundancy.
```

- **用「複製」而不是「搬移」**：原本的文字 chunk 仍然保留這幾行，資訊不會遺失。
- **只抓有用到的數字**：避免把一般的編號清單（例如「1. 拆下外殼」）誤當成註腳。

### 5. 表格也記錄章節

表格 chunk 記錄**它出現時所在的章節**，所以可以知道這張表屬於「ELECTRICAL SPECIFICATIONS」。

## 五、每個 chunk 的前綴

每個 chunk 的第一行：

```
[Advanced Energy | CSU2400AT | ae-csu2400at.pdf | p.2 | ELECTRICAL SPECIFICATIONS]
```

這一行同時解決三個問題：

| 用途 | 說明 |
|---|---|
| **檢索** | 內文常寫 "the power supply"，而不寫型號。有了前綴，問「TEC3200 的 hold-up time」時，BM25 才比對得到 `TEC3200`，向量裡也會帶有型號資訊 |
| **讓模型理解** | 模型同時拿到 5 個 chunk 時，知道哪個數字屬於哪個型號，不會把 TEC1600 的值套到 TEC3200（黃金樣本的 `similar` 題就是在考這個） |
| **引用出處** | 回答時附上的檔名和頁碼，直接從這裡取得 |

頁碼可能是範圍：chunk 跨頁時寫成 `p.12-14`。

## 六、chunk 的欄位

```python
Chunk(
    chunk_id, file, vendor, model,   # 身分
    kind,                            # "text" 或 "table"
    page, page_end, section,         # 位置
    text,                            # 前綴 + 內文：真正被索引、給模型看的內容
    labels_inferred,                 # 表格的參數名稱是推測的嗎？（品質旗標）
    doc_sha256,                      # 來自哪個版本的 PDF
    revision,                        # 預留：文件版本
    access_level,                    # 預留：權限，預設 "public"
)
```

- **為什麼現在就預留 `revision` 和 `access_level`？** 如果之後要做權限過濾，才發現索引裡沒有這個欄位，就得整份重建索引。現在先放預設值，之後只要填值。
- **前綴已經有 `page` 和 `section`，為什麼還要獨立欄位？** 前綴是給**文字比對和模型**看的；獨立欄位是給**程式**用的，例如過濾「只搜 TEC3200」、產生引用。

## 七、chunk_id：用內容決定 ID

```python
chunk_id = sha256(file + "\n" + text)[:16]
```

- 內容相同，ID 永遠相同：重跑 100 次結果都一樣。
- 內容一改，ID 就跟著改。

**用途**：下一步做 embedding 時，以 chunk_id 當快取的 key，已經算過的就跳過。例如修好 LMS2700 的解析後，只有它的 chunk 換了 ID，其他 600 多個都直接用快取。

**為什麼不用流水號（1、2、3…）？** 前面只要多出一個 chunk，後面的編號全部位移，快取跟著失效；更糟的是，同一個 ID 前後會指向不同的內容。

最後還會去重：ID 相同代表同一頁上有完全相同的內容，只保留一份。

## 八、解析快取

解析 9 份 PDF 要約 2 分鐘，切段不到 1 秒，所以把**解析結果**存起來。key 是：

```
PDF 的 SHA-256 前 16 碼 + PARSER_VERSION
→ data/processed/parsed/bel-tec1600.279457e9a1f556aa.v1.json
```

| 情況 | 結果 |
|---|---|
| PDF 換成新版 | 雜湊值不同 → 重新解析 |
| **修改了 parse.py** | PDF 沒變，雜湊值也一樣 → **必須手動把 `PARSER_VERSION` 加一**，否則會拿到舊程式產生的過期結果 |
| 都沒有變 | 直接讀快取，約 0.3 秒 |

寫入時先寫到暫存檔再改名，即使中途當掉，也不會留下寫到一半的快取（和下載 PDF 的做法一樣）。同一份 PDF 的舊快取檔會一併刪除。

## 九、怎麼知道切得好不好

`tests/test_chunk.py` 分成兩層。

**單元測試**（使用自己造的小資料，不需要 PDF）：

| 測試 | 驗證什麼 |
|---|---|
| `test_dedup_row_merges_colspan_copies` | colspan 去重 |
| `test_big_table_is_split_by_rows_and_repeats_header` | 大表切開後每一段都有表頭，**而且沒有任何一列遺失** |
| `test_footnotes_found_in_any_list_style` | `1.` 和 `- 2` 都抓得到，沒用到的 `3.` 不會被抓 |
| `test_table_without_markers_gets_no_footnotes` | 沒有記號的表格不會被附上註腳 |
| `test_headings_set_section_and_prefix` | 標題設定章節，前綴格式正確 |
| `test_long_section_is_split_with_overlap` | 長段落會被切開，下一段的開頭是上一段的結尾 |
| `test_overlap_does_not_cross_headings` | overlap 不跨章節 |
| `test_chunk_ids_are_stable_and_content_based` | 同樣的內容得到同樣的 ID，不同的內容得到不同的 ID |

**整合測試**（真實 PDF 加上黃金樣本）：`test_golden_answer_is_in_source_page_chunks`

每一題都檢查：**答案有沒有出現在來源頁的 chunk 裡**。目前 40 題通過，2 題是已知的解析問題（標為 xfail）。

這個測試量的是 **RAG 的上限**：答案根本不在 chunk 裡的話，檢索做得再好也找不到。之後評測分數低時，可以先看這個測試：

- 這個測試**失敗** → 問題出在切段或解析；
- 這個測試**通過**，但回答錯誤 → 問題出在檢索或生成。

比對時有兩個細節：

- 表格把數值和單位放在不同欄（`| 740 | g |`），所以先拿掉 `|` 再比對，`740g` 才比對得到。
- 只檢查 `must_include`，不檢查 `must_not_include`。後者是用來檢查**回答**的：同一頁本來就可能列出其他負載下的 94%，那不代表切錯。

## 十、目前的限制

1. **token 是估算的**：以「字元數 ÷ 4」估算。選定 embedding 模型之後，應該改用它的 tokenizer 計算。
2. **有 59 個很小的 chunk**（少於 40 tokens），多半是表格標題，例如「PMBus Command List:」。把它們併進下一張表會更好，但要不要做，交給評測結果決定。
3. **跨頁大表切開後，每一段標的都是整張表的頁碼範圍**，因為合併表格時沒有保留每一列所在的頁碼。
4. **註腳是用規則判斷的**：格式特殊的文件可能會漏抓。

## 十一、常見問題（Q&A）

### Q：chunk 大小怎麼決定的？

400 是常見的起點，最後要用黃金樣本的 recall@k 比較 200、400、800 才算數。表格放寬是因為切開的代價比較大。

### Q：為什麼不用 LangChain 的 RecursiveCharacterTextSplitter？

它只看字元，不知道哪裡是表格、表頭和註腳。規格書的答案大多在表格裡，通用切法正好會在最重要的地方出錯。

### Q：overlap 會不會讓同一段內容被檢索兩次？

會，所以之後要對檢索結果去重，或者合併相鄰的 chunk。代價是多約 15% 的儲存空間和 embedding 成本。

### Q：文件更新時怎麼辦？

chunk_id 由內容決定，所以只需要重新 embedding 有變動的 chunk。舊版的 chunk 要從索引中刪掉，詳見 README 路線圖的「文件版本」。
