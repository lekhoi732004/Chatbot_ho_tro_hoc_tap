from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "edu_agent_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


app = FastAPI(title="Edu Agent Chat UI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    file_name: str
    file_type: str
    text_length: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


@app.get("/")
async def read_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "edu-agent-chat-ui"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    from src.graph.workflow import run_workflow

    try:
        result = await asyncio.to_thread(
            run_workflow,
            request.message.strip(),
            request.session_id,
        )
        if result is None:
            raise RuntimeError("Workflow không trả về kết quả")
        if not isinstance(result, dict):
            result = {"answer": str(result), "metadata": {}}

        reply = str(result.get("answer") or "")
        return ChatResponse(
            reply=reply,
            status="success",
            metadata=result.get("metadata") or {},
        )
    except Exception as exc:
        return ChatResponse(
            reply=f"Lỗi hệ thống: {exc}",
            status="error",
            metadata={},
        )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form(default="default")
):
    """
    Ingest a document file (PDF, DOCX, TXT, etc.)
    Extract text and add to vector database + memory.
    
    Args:
        file: Uploaded file (pdf, docx, txt, md, csv, etc.)
        session_id: Session to associate file with
    
    Returns:
        IngestResponse with extraction result and metadata
    """
    file_name = file.filename or "không_rõ_tên"
    file_ext = Path(file_name).suffix.lower()
    
    # Supported file types
    supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".rst", ".csv"}
    
    if file_ext not in supported_extensions:
        return IngestResponse(
            status="error",
            file_name=file_name,
            file_type=file_ext,
            text_length=0,
            error=f"Định dạng file '{file_ext}' chưa được hỗ trợ. Các định dạng hỗ trợ: {', '.join(sorted(supported_extensions))}"
        )
    
    temp_file_path = None
    try:
        # Save uploaded file to temp location
        temp_file_path = UPLOAD_DIR / file_name
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Extract document using router
        from src.tools.router import extract_document
        from src.memory.memory_manager import get_memory_manager
        
        text, metadata = await asyncio.to_thread(
            extract_document,
            str(temp_file_path)
        )
        
        if not text or len(text.strip()) == 0:
            return IngestResponse(
                status="error",
                file_name=file_name,
                file_type=file_ext,
                text_length=0,
                error="Không trích xuất được văn bản từ file. File có thể rỗng hoặc bị lỗi."
            )

        upload_metadata = {
            "source_file": file_name,
            "file_name": file_name,
            "file_type": file_ext.lstrip("."),
            "source_origin": "user_upload",
            "uploaded_file": True,
            "session_id": session_id,
            "upload_file_name": file_name,
            "retrieval_priority": "user_upload",
        }
        
        # Save to memory with file context
        memory_manager = get_memory_manager()
        file_msg = await asyncio.to_thread(
            memory_manager.add_file_message,
            session_id,
            file_name,
            metadata.get("type", file_ext.lstrip(".")),
            text,
            metadata={
                "source_file": file_name,
                "chunks_created": metadata.get("pages_extracted", 
                                              metadata.get("headings", 0) + metadata.get("tables", 0)),
                **upload_metadata,
                **metadata
            }
        )
        
        # Ingest uploaded files into the same hybrid vectorstore used by data_main.
        try:
            from src.rag.pipeline import IngestPipeline
            await asyncio.to_thread(
                lambda: IngestPipeline(force_reingest=True)
                .load()
                .ingest_file(str(temp_file_path), extra_metadata={**upload_metadata, **metadata})
            )
            try:
                import src.rag.retriever as retriever_module
                retriever_module._vectorstore = None
            except Exception:
                pass
        except Exception as ingest_exc:
            # Log warning but still return success since extraction worked.
            import logging
            logging.warning(f"Ingest to vector DB failed: {ingest_exc}")
            metadata["ingest_warning"] = str(ingest_exc)
        
        return IngestResponse(
            status="success",
            file_name=file_name,
            file_type=metadata.get("type", file_ext.lstrip(".")),
            text_length=len(text),
            metadata={
                "chunks_created": metadata.get("pages_extracted", 
                                              metadata.get("headings", 0) + metadata.get("tables", 0)),
                "source_type": metadata.get("type"),
                "saved_to_memory": True,
                "msg_id": file_msg.msg_id,
                **upload_metadata,
                **metadata
            }
        )
        
    except Exception as exc:
        import traceback
        return IngestResponse(
            status="error",
            file_name=file_name,
            file_type=file_ext,
            text_length=0,
            error=f"Xử lý file thất bại: {str(exc)}\n{traceback.format_exc()}"
        )
    finally:
        # Clean up temp file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    uvicorn.run(
        "khung_chat_bot.scr.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
