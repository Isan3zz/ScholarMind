from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from app.graph.graph import create_graph
import json
import asyncio
import os
import shutil
from app.rag.engine import process_documents, reset_knowledge_base, UPLOAD_DIR
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.trace.collector import TraceCollector, TraceTurn
from app.trace.storage import load_trace, save_trace

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "checkpoints.db")
router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    search_mode: str = "hybrid"
    thread_id: str
    intent: str | None = None


@router.post("/clear")
async def clear_endpoint():
    try:
        reset_knowledge_base()
        return {"message": "知识库已重置", "status": "success"}
    except Exception as e:
        print(f"清空失败: {e}")
        return {"message": f"清空失败: {str(e)}", "status": "error"}


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="一次最多只能上传 5 个文件")

    try:
        reset_knowledge_base()
        saved_paths = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)

        chunks_num = await asyncio.to_thread(process_documents, saved_paths)
        return {
            "status": "success",
            "file_count": len(files),
            "chunks_stored": chunks_num,
            "message": "文档解析完成，知识库构建成功",
        }
    except Exception as e:
        print(f"上传处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    thread_id = request.thread_id

    async def event_generator():
        existing = load_trace(thread_id)
        collector = TraceCollector(thread_id)
        if existing:
            collector._started_at = existing.get("started_at", collector._started_at)
            collector.turns = _rehydrate_turns(existing.get("turns", []))

        turn = collector.begin_turn(
            query=request.query,
            intent=request.intent,
            mode=request.search_mode,
        )

        print(f"[Chat] mode={request.search_mode} query={request.query}")

        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
            app = create_graph(memory=memory)

            initial_state = {
                "query": request.query,
                "revision_number": 0,
                "search_mode": request.search_mode,
                "should_stop": False,
                "intent": request.intent,
            }

            async for event in app.astream(initial_state, config=config):
                for node_name, state_update in event.items():
                    collector.record_node(turn, node_name, state_update)
                    data = json.dumps(
                        {"step": node_name, "data": state_update}, ensure_ascii=False
                    )
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.1)

            # Persist short_memory to checkpoint (post-processing, not a graph node)
            final_state = await app.aget_state(config)
            if final_state and final_state.values:
                from app.graph.nodes.memory import update_short_memory

                mem_update = update_short_memory(final_state.values)
                await app.aupdate_state(config, mem_update)

        collector.finish_turn(turn)
        path = save_trace(collector)
        print(f"--- [Trace] 已保存 {thread_id} → {path} ({len(collector.turns)} turn(s))")

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _rehydrate_turns(data: list[dict]) -> list:
    turns = []
    for item in data:
        t = TraceTurn(
            turn=int(item.get("turn", 0)),
            query=str(item.get("query", "")),
            intent=item.get("intent"),
            mode=str(item.get("mode", "")),
            started_at=str(item.get("started_at", "")),
            finished_at=str(item.get("finished_at", "")),
        )
        t.plan = list(item.get("plan", []))
        t.evidence = list(item.get("evidence", []))
        t.report = str(item.get("report", ""))
        review = item.get("review", {}) or {}
        t.review_status = str(review.get("status", ""))
        t.critique = str(review.get("critique", ""))
        t.revision_number = int(review.get("revision", 0))
        t.should_stop = bool(review.get("should_stop", False))
        t.nodes = list(item.get("nodes", []))
        turns.append(t)
    return turns
