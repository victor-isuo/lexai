"""
Legal Eval — LexAI
====================
LLM-as-judge evaluation for contract analysis quality.

Three scoring axes:
1. Clause Detection Accuracy (40%) — did the system find all significant clauses?
2. Legal Grounding Accuracy (35%) — are findings cited and legally sound?
3. Recommendation Specificity (25%) — are recommendations actionable?

Same LLM-as-judge pattern as AgentEval — this is the signature.
Every LexAI output is scoreable, not just claimed to be good.
"""

import os
import time
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from groq import Groq

logger = logging.getLogger(__name__)
PASS_MARK = 0.70
LOG_PATH = Path("data/eval_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Judge call ────────────────────────────────────────────────────────────────

def _call_judge(prompt: str) -> str:
    """Call Groq Llama 4 as the evaluation judge.
    Groq is fine here — single short call, not a multi-agent chain.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    res = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=600,
    )
    return res.choices[0].message.content


def _parse_scores(raw: str, axes: list[str]) -> dict:
    """Extract JSON scores from judge response."""
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        return {ax: float(parsed.get(ax, 0.5)) for ax in axes}
    except Exception:
        return {ax: 0.5 for ax in axes}


# ── Evaluation axes ───────────────────────────────────────────────────────────

AXES = ["clause_detection", "legal_grounding", "recommendation_specificity"]
WEIGHTS = {"clause_detection": 0.40, "legal_grounding": 0.35, "recommendation_specificity": 0.25}

JUDGE_PROMPT = """You are an expert legal AI evaluator assessing the quality of a contract analysis report.

CONTRACT TYPE: {contract_type}
CONTRACT EXCERPT (first 1500 chars):
{contract_excerpt}

ANALYSIS REPORT:
{report}

Score the report on exactly these three axes. Return ONLY a JSON object with scores from 0.0 to 1.0:

clause_detection (0.0-1.0):
  Did the analysis identify all significant clause types present in the contract?
  Did it miss any important clauses (limitation of liability, indemnity, termination, IP, etc.)?
  1.0 = comprehensive clause identification, 0.0 = missed most significant clauses

legal_grounding (0.0-1.0):
  Are the findings grounded in specific legal principles or standards?
  Does the report cite legal frameworks (Common Law, UNIDROIT, specific standards)?
  Are risk assessments legally justified rather than generic?
  1.0 = every finding legally grounded with citation, 0.0 = pure assertion with no legal basis

recommendation_specificity (0.0-1.0):
  Are the recommendations specific and actionable?
  Does the report say WHAT to negotiate, not just that something is risky?
  Could a business person act on these recommendations without further guidance?
  1.0 = every recommendation specific and immediately actionable, 0.0 = vague warnings only

After the JSON, write one sentence explaining the lowest score.

Return format:
{{"clause_detection": 0.0, "legal_grounding": 0.0, "recommendation_specificity": 0.0}}
Reasoning: [one sentence]"""


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_analysis(
    contract_text: str,
    contract_type: str,
    analysis_report: str,
    filename: str = "contract.pdf",
) -> dict:
    """
    Evaluate a LexAI contract analysis report.

    Args:
        contract_text: Full contract text
        contract_type: Detected contract type
        analysis_report: The report produced by the crew
        filename: Original filename

    Returns:
        Eval result dict with score, passed, axis_scores, reasoning
    """
    logger.info(f"Evaluating analysis for: {filename}")

    prompt = JUDGE_PROMPT.format(
        contract_type=contract_type,
        contract_excerpt=contract_text[:1500],
        report=analysis_report[:3000],
    )

    try:
        raw = _call_judge(prompt)
        axis_scores = _parse_scores(raw, AXES)
        final_score = sum(axis_scores[ax] * WEIGHTS[ax] for ax in AXES)
        passed = final_score >= PASS_MARK

        reasoning = ""
        if "Reasoning:" in raw:
            reasoning = raw.split("Reasoning:")[-1].strip()

        worst_axis = min(axis_scores, key=axis_scores.get)

        result = {
            "eval_id": str(uuid.uuid4())[:8],
            "filename": filename,
            "contract_type": contract_type,
            "score": round(final_score, 3),
            "passed": passed,
            "axis_scores": axis_scores,
            "worst_axis": worst_axis,
            "reasoning": reasoning,
            "evaluated_at": datetime.utcnow().isoformat(),
        }

        _log_result(result)
        return result

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            "eval_id": "error",
            "filename": filename,
            "contract_type": contract_type,
            "score": 0.0,
            "passed": False,
            "axis_scores": {ax: 0.0 for ax in AXES},
            "worst_axis": "error",
            "reasoning": str(e),
            "evaluated_at": datetime.utcnow().isoformat(),
        }


def _log_result(result: dict):
    """Append eval result to JSONL log file."""
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(result) + "\n")
    except Exception as e:
        logger.warning(f"Eval log write failed: {e}")


def load_eval_history(limit: int = 20) -> list[dict]:
    """Load recent eval results from log file."""
    if not LOG_PATH.exists():
        return []
    results = []
    try:
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results[-limit:]
    except Exception:
        return []


def eval_summary(history: list[dict]) -> dict:
    """Compute summary stats from eval history."""
    if not history:
        return {"total": 0, "passed": 0, "avg_score": None, "pass_rate": None}
    scores = [r["score"] for r in history]
    passed = sum(1 for r in history if r["passed"])
    return {
        "total": len(history),
        "passed": passed,
        "failed": len(history) - passed,
        "avg_score": round(sum(scores) / len(scores), 3),
        "pass_rate": round(passed / len(history), 3),
    }