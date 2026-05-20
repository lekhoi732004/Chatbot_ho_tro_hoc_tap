# edu-agent

Chatbot giáo dục chạy local với LangGraph, RAG, memory, classifier định tuyến, optimizer prompt, executor sinh câu trả lời và verifier kiểm duyệt.

## Luồng Hệ Thống

```text
Người dùng
  -> Classifier: xác định intent, loại nhiệm vụ, chủ đề, subtasks và kiểm tra đúng phạm vi
  -> RAG: lấy ngữ cảnh từ tài liệu khi cần
  -> Optimizer: ghép prompt cuối từ classifier, RAG và memory
  -> Executor: sinh câu trả lời bằng Qwen2.5 3B
  -> Verifier: kiểm duyệt câu trả lời bằng Qwen2.5 0.5B
  -> Retry: nếu verifier không đạt, đưa feedback về executor xử lý lại
  -> Memory: lưu hội thoại
```

Classifier sẽ chặn câu hỏi ngoài phạm vi học tập/chuyên môn trước khi sinh câu trả lời.

## Yêu Cầu

- Python 3.10+
- RAM tối thiểu 8 GB, khuyến nghị 16 GB
- GPU CUDA khuyến nghị nếu chạy Qwen2.5 3B

Cài thư viện:

```bash
pip install -r requirements.txt
```

Trên Windows, dùng đúng Python executable đang cài trên máy:

```powershell
py -m pip install -r requirements.txt
```

## Môi Trường

Tạo `.env` từ `.env.example`:

```bash
cp .env.example .env
```

LangSmith tracing là tùy chọn:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=chatbot_edu
```

`.env` đã được đưa vào `.gitignore` vì có thể chứa secret.

## Tải Model

```bash
python scripts/download_models.py
```

Các đường dẫn model trong `config/config.json`:

- `llm.executor`: `models/llm/qwen2.5-3b-instruct`
- `llm.classifier`: `models/llm/qwen2.5-0.5b-instruct`
- `llm.optimizer`: `models/llm/qwen2.5-0.5b-instruct`
- `llm.verifier`: `models/llm/qwen2.5-0.5b-instruct`
- embedding: `models/embedding/bge-base-en-v1.5`
- reranker: `models/reranker/bge-reranker-base`

## Tạo Vector Database

RAG database mặc định được build từ thư mục `data_main`.

Thành phần:

- FAISS `IndexIVFFlat` với HNSW quantizer cho semantic search
- BM25 cho tìm kiếm từ khóa
- Hybrid retrieval kết hợp vector score và BM25 score
- Cross-encoder reranker để xếp hạng lại candidate trước khi đưa vào context

Build database:

```bash
python scripts/build_vector_database.py --clear
```

Trên Windows trong môi trường Conda hiện tại:

```powershell
D:\App\Anaconda\data\envs\chatbot3\python.exe scripts\build_vector_database.py --clear
```

GPU:

- `config/config.json` đang đặt `embedding.device`, `reranker.device` và `rag.faiss.device` là `cuda`.
- Embedding/reranker dùng GPU nếu PyTorch thấy CUDA.
- FAISS chỉ dùng GPU nếu môi trường Python cài bản FAISS có GPU API. Nếu đang dùng `faiss-cpu`, hệ thống tự fallback sang CPU cho FAISS index và vẫn dùng GPU cho embedding.

Lệnh tương đương:

```bash
python scripts/ingest.py --path data_main --clear
```

Xem thống kê:

```bash
python scripts/ingest.py --show-stats
```

File database nằm trong `vectorstore/faiss_index`:

- `index.faiss`
- `metadata.pkl`
- `chunks.jsonl`
- `bm25.pkl`
- `manifest.json`
- `hashes.pkl`

Test retrieval có reranker:

```bash
python scripts/extract.py search "overfitting là gì" --top-k 5 --rerank
```

## Chạy CLI

Chat tương tác:

```bash
python -m src.main
```

Một câu hỏi:

```bash
python -m src.main --query "Giải thích RAG là gì" --session demo
```

## Chạy Giao Diện Web

Giao diện nằm trong `khung_chat_bot` và gọi cùng LangGraph workflow.

Khởi động server từ thư mục gốc repo:

```bash
uvicorn khung_chat_bot.scr.server:app --host 127.0.0.1 --port 8000 --reload
```

Mở:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Test API chat:

```bash
curl -X POST http://127.0.0.1:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Giải thích machine learning là gì\",\"session_id\":\"demo\"}"
```

Trên macOS/Linux, thay `^` bằng `\`.

## LangSmith Tracing

Khi `LANGSMITH_TRACING=true`, workflow sẽ được trace với:

- run name: `edu_agent_workflow`
- tags: `edu-agent`, `langgraph`, `session:<session_id>`
- node được trace: classify, optimize, execute, verify, retry, finalize

Trace được gửi đến project trong `LANGSMITH_PROJECT`.

## Cấu Trúc Chính

```text
edu-agent/
  config/config.json
  data_main/
  khung_chat_bot/
    scr/server.py
    static/index.html
    static/script.js
    static/style.css
  src/
    agents/
      classifier.py
      optimizer.py
      executor.py
      verifier.py
    graph/workflow.py
    memory/
    rag/
      vectorstore.py
      retriever.py
      reranker.py
    tools/
    utils/
      tracing.py
      llm.py
      logger.py
  scripts/
    build_vector_database.py
    download_models.py
    ingest.py
  vectorstore/faiss_index/
```

## Kiểm Tra Nhanh

Validate config:

```powershell
Get-Content -Raw config/config.json | ConvertFrom-Json | Out-Null
```

Compile Python:

```bash
python -m compileall src khung_chat_bot scripts
```

Start UI và kiểm tra:

```bash
uvicorn khung_chat_bot.scr.server:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/api/health
```
