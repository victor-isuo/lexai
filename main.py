"""
LexAI — AI Contract Intelligence
==================================
FastAPI application.

Endpoints:
  GET /ui — Web interface
  POST /analyze — Upload and analyse contract PDF
  GET /eval/history — Recent evaluation results
  GET /eval/summary — Evaluation summary stats
  GET /health — System health
  GET /docs — Swagger
"""

import os
import uuid
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

knowledge_base = None
is_ready = False


async def initialise():
    global knowledge_base, is_ready
    try:
        logger.info("Building LexAI knowledge base...")
        from src.core.knowledge_base import build_knowledge_base
        from src.tools.retrieval_tool import set_vectorstore
        knowledge_base = build_knowledge_base()
        set_vectorstore(knowledge_base)
        is_ready = True
        logger.info("LexAI ready.")
    except Exception as e:
        logger.error(f"Initialisation failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(initialise())
    yield
    logger.info("LexAI shutting down.")


app = FastAPI(
    title="LexAI",
    description="AI-powered contract analysis grounded in Common Law principles",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
Path("data/uploads").mkdir(parents=True, exist_ok=True)


@app.get("/ui", include_in_schema=False)
async def ui():
    return FileResponse("static/index.html")


@app.get("/")
async def root():
    return {
        "name": "LexAI",
        "version": "1.0.0",
        "status": "ready" if is_ready else "initialising",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ready": is_ready,
        "knowledge_base": knowledge_base is not None,
    }


@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    """Upload a contract PDF and receive a structured analysis report."""
    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail="LexAI is still initialising. Please wait 30 seconds and try again."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_id = str(uuid.uuid4())[:8]
    save_path = Path("data/uploads") / f"{file_id}_{file.filename}"

    try:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum 10MB.")

        save_path.write_bytes(content)

        # Parse
        from src.core.document_parser import ContractParser
        parsed = ContractParser().parse(str(save_path))

        if not parsed.full_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from PDF. "
                       "Ensure the PDF contains selectable text, not scanned images."
            )

        metadata = {
            "contract_type":  parsed.metadata.contract_type,
            "governing_law":  parsed.metadata.governing_law,
            "parties":        parsed.metadata.parties,
            "effective_date": parsed.metadata.effective_date,
            "total_pages":    parsed.metadata.total_pages,
        }

        # Analyse
        from src.agents.legal_crew import analyse_contract
        result = analyse_contract(
            contract_text=parsed.full_text,
            contract_metadata=metadata,
            filename=file.filename,
        )

        # Build response first — eval is non-blocking
        response = {
            "filename":         file.filename,
            "contract_type":    result["contract_type"],
            "overall_risk":     result["overall_risk"],
            "analysis_report":  result["analysis_report"],
            "processing_time":  result["processing_time"],
            "clauses_detected": len(parsed.clauses),
            "pages":            parsed.metadata.total_pages,
            "governing_law":    parsed.metadata.governing_law,
            "parse_warnings":   parsed.parse_warnings,
        }

        # Evaluate if enabled
        if os.getenv("ENABLE_EVAL", "false").lower() == "true":
            try:
                from src.evaluation.legal_eval import evaluate_analysis
                eval_result = evaluate_analysis(
                    contract_text=parsed.full_text,
                    contract_type=result["contract_type"],
                    analysis_report=result["analysis_report"],
                    filename=file.filename,
                )
                response["eval"] = {
                    "score":       eval_result["score"],
                    "passed":      eval_result["passed"],
                    "axis_scores": eval_result["axis_scores"],
                }
                logger.info(
                    f"Eval — Score: {eval_result['score']:.3f} "
                    f"| Passed: {eval_result['passed']}"
                )
            except Exception as e:
                logger.warning(f"Eval failed (non-blocking): {e}")

        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/eval/history")
async def eval_history(limit: int = 20):
    """Recent evaluation results."""
    from src.evaluation.legal_eval import load_eval_history
    history = load_eval_history(limit=limit)
    return {"history": history, "total": len(history)}


@app.get("/eval/summary")
async def eval_summary_endpoint():
    """Evaluation summary statistics."""
    from src.evaluation.legal_eval import load_eval_history, eval_summary
    history = load_eval_history(limit=100)
    return eval_summary(history)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


