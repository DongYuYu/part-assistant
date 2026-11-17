# Kubernetes（kind）

在本機用 [kind](https://kind.sigs.k8s.io/) 跑一個 Kubernetes 叢集，部署推論閘道：3 個副本、共用 Redis 的速率限制、健康檢查、從 Secret 讀 API key。kind 是用 Docker 容器扮演節點，所以不需要真的機器，但裡面的 Kubernetes API 和正式環境（例如 EKS）一樣。

## 檔案

| 檔案 | 內容 |
|---|---|
| `kind-cluster.yaml` | 叢集：1 個節點，Kubernetes v1.37.0（節點映像檔用 digest 鎖定） |
| `namespace.yaml` | 全部放在 `part-assistant` 這個 namespace，方便查看和刪除 |
| `redis.yaml` | Redis（Deployment + Service）：速率計數器，只放記憶體 |
| `gateway.yaml` | 閘道：PVC（SQLite）、Deployment（3 個副本、健康檢查、資源上限、非 root）、Service |
| `ollama.yaml` | `ExternalName` Service：讓叢集裡用 `ollama` 這個名字連到 Compose 裡的 Ollama |
| `kustomization.yaml` | `kubectl apply -k k8s/` 一次部署上面全部 |

## 步驟

需要 Docker、[kind](https://github.com/kubernetes-sigs/kind/releases) v0.33.0、kubectl v1.37.0。

```bash
# 1. 叢集
kind create cluster --config k8s/kind-cluster.yaml

# 2. 映像檔：kind 的節點看不到本機 Docker 的映像檔，要複製進去（正式環境是推到 registry）
docker compose build gateway
kind load docker-image part-assistant-gateway:local --name part-assistant

# 3. 地端模型：Ollama 跑在 Compose，接到 kind 的 Docker 網路，叢集裡才連得到
docker compose up -d ollama
docker network connect kind liteon-agent-ollama-1

# 4. Secret（真的 API key，不進 git），再部署全部
kubectl apply -f k8s/namespace.yaml
grep '^ANTHROPIC_API_KEY=' .env > /tmp/key.env
kubectl -n part-assistant create secret generic gateway-secrets --from-env-file=/tmp/key.env
rm /tmp/key.env
kubectl apply -k k8s/
kubectl -n part-assistant rollout status deploy/gateway

# 5. 發金鑰（寫進共用的 SQLite，3 個 Pod 都認得）
kubectl -n part-assistant exec deploy/gateway -- python -m part_assistant.gateway add-key alice

# 清除
kind delete cluster --name part-assistant
docker network disconnect kind liteon-agent-ollama-1
```

## 實測（2026-11-17，WSL2，kind v0.33.0）

1. **Service 分配請求，3 個 Pod 共用速率限制**：
   - 金鑰設定每分鐘 4 次，從叢集裡的 curl Pod 送 8 個請求給 `gateway:8100`。
   - 結果是 4 個 200、4 個 429，請求分散到 3 個 Pod。
   - 其中一個 Pod 只收到 429：它自己沒有放行過任何請求，但從 Redis 知道已經滿了。
2. **自動補回**：刪掉一個閘道 Pod，1 秒內就有新的 Pod 取代它，Deployment 一直是 3/3。

## 限制

- **只有 1 個節點**：3 個 Pod 共用一個 ReadWriteOnce 的磁碟（SQLite），只有在同一個節點時才行。多個節點要把金鑰和用量改到 Postgres，正式環境通常用託管資料庫（RDS），叢集裡只放連線字串的 Secret。
- **1 個節點不能防機器故障**：筆電一關，全部都停。這裡學的是 Kubernetes 的操作（自動補回、負載平衡、健康檢查），可用性要靠多個節點。
- **沒有對外入口**：只能從叢集裡呼叫，或用 `kubectl port-forward`。正式環境會用 Ingress 或雲端的 Load Balancer，加上 TLS。
- **地端模型在叢集外**：kind 很難使用 GPU。正式環境是 vLLM 跑在 GPU 節點上，用一般的 Service 連線。
