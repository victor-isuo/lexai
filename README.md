---
title: LexAI
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# ⚖️ LexAI — AI Contract Intelligence

AI-powered contract analysis grounded in Common Law principles and UNIDROIT international standards. Upload any contract PDF and receive a structured risk report with per-clause findings, severity ratings, and specific recommendations.

## 🔴 Live Demo

[https://victorisuo-lexai.hf.space/ui](https://victorisuo-lexai.hf.space/ui)

---

## What This Is

Most AI contract tools are RAG chatbots — they retrieve text and summarise it. LexAI is different. It applies a four-agent reasoning pipeline where each specialist hands off to the next: clause extraction → legal research → risk assessment → plain language report. Every finding is grounded in a curated legal knowledge base. Every risk rating is justified against an established legal standard.

The difference between retrieving contract text and reasoning about it against established law is the difference between a keyword search and a legal opinion.

---

## System Architecture

```
Contract PDF Upload
        ↓
Document Parser
(clause boundary detection · type classification
 automated risk signal scanning · metadata extraction)
        ↓
Clause-Driven Input
(structured ContractClause objects, priority-ordered
 by risk signals — not raw text truncation)
        ↓
ChromaDB Legal Knowledge Base
(Common Law principles · clause standards
 UNIDROIT standards · risk frameworks)
        ↓
CrewAI Four-Agent Sequential Pipeline
        ↓
┌────────────────────────────────────────────┐
│  Senior Contract Analyst                   │
│  Reviews pre-extracted clauses,            │
│  confirms types, identifies obligations    │
│                    ↓                       │
│  Legal Research Specialist                 │
│  Grounds each clause in the knowledge base │
│  with citations                            │
│                    ↓                       │
│  Contract Risk Assessment Specialist       │
│  Scores CRITICAL / HIGH / MEDIUM / LOW     │
│  with legal justification per clause       │
│                    ↓                       │
│  Legal Plain Language Specialist           │
│  Structured report with specific actions   │
└────────────────────────────────────────────┘
        ↓
Structured Analysis Report
(risk summary · per-clause findings
 legal citations · recommended actions)
        ↓
LLM-as-Judge Evaluation
(clause detection · legal grounding
 recommendation specificity · pass/fail score)
```

---

## What Makes This Different From Generic RAG

**Clause-aware parsing before the agents run.** The `ContractParser` identifies clause boundaries, classifies each clause by type (liability, indemnity, termination, IP etc.), and scans for automated risk signals before the LLM reasoning step begins. The agents receive structured `ContractClause` objects — not raw character slices. A contract with 30 clauses has its highest-risk clauses selected first, not its first 6,000 characters.

**Four-agent role handoff.** Senior Contract Analyst → Legal Research Specialist → Contract Risk Assessment Specialist → Legal Plain Language Specialist. Each agent has a defined role, a specific tool set, and a backstory that anchors its reasoning. The Legal Researcher cannot produce findings without querying the knowledge base. The Risk Assessor cannot score without the Researcher's citations. The pipeline enforces analytical discipline a single-agent system cannot replicate.

**Pre-flight risk signal detection.** Fourteen risk patterns — unlimited liability, broad indemnity, unilateral modification rights, automatic renewal, one-sided termination, broad IP assignment, non-compete clauses — are detected automatically on every clause by regex pattern matching before any LLM call is made. These signals are passed to the agents as structured input, not discovered mid-reasoning.

**LLM-as-judge evaluation.** Every analysis is independently scored by a separate judge model on three axes: clause detection accuracy (40%), legal grounding accuracy (35%), and recommendation specificity (25%). The score is returned with every API response. This is not claimed accuracy — it is measured accuracy.

---

## ⚠️ Current Limitations — Free Tier Infrastructure

This is an honest account of what the demo does and does not do. Understanding these constraints matters for evaluating the system correctly.

### Infrastructure Constraints

The live demo runs on **Groq free tier** (30,000 tokens/minute) with **Hugging Face Spaces free CPU**.

| Constraint | Demo | Production |
|------------|------|------------|
| LLM | Groq Llama-4-Scout 17B (free) | GPT-4o / Claude 3.5 Sonnet / Groq paid |
| Token budget | ≤ 27,000 tokens/run | Uncapped |
| Clause input | Top 12 clauses by risk priority | All clauses, full text |
| Chars per clause | 600 characters | 3,000+ characters |
| Processing time | 90–180 seconds | 30–60 seconds |
| Concurrent users | 1 (rate limit shared) | Scales with infrastructure |

### Model Constraints

**Llama-4-Scout 17B has a practical reasoning limit** that becomes visible on complex contracts. The four-agent sequential pipeline accumulates context across tasks — by Task 3, the model is carrying 15,000+ tokens of prior reasoning. At this scale, a 17B parameter model on free tier will:

- Truncate clause coverage, typically completing 5–8 clauses before concluding
- Lose citation fidelity — legal source names from the KB become less precise
- Produce shallower recommendations on later clauses than earlier ones
- Occasionally return empty responses under sustained load, triggering the retry/fallback chain

**This is a model capability constraint, not an architectural one.** The same pipeline on GPT-4o or Claude 3.5 Sonnet processes all clauses with consistent depth and full citation fidelity across the entire contract.

### What the Demo Proves vs What It Does Not

**What it proves:**
- The full pipeline runs end-to-end: parse → clause extraction → 4-agent reasoning → KB retrieval → risk scoring → structured report → LLM-as-judge evaluation
- The architecture is sound — clause-driven input, structured agent handoffs, grounded output format, independent evaluation
- The engineering decisions are production-grade: fallback LLM chain, TPM-aware retry logic, structured data throughout, measurable eval scores

**What it does not prove on free tier:**
- Complete clause coverage for contracts longer than ~8 clauses
- Consistent citation depth across all clauses
- Production throughput or concurrent request handling

### Production Path

Two changes move this to full production capability:

```python
# legal_crew.py
MAX_CLAUSES     = None   # process all clauses
CLAUSE_CHAR_CAP = 3000   # full clause text

# .env
GROQ_API_KEY = <paid tier key>
# or substitute: OPENAI_API_KEY / ANTHROPIC_API_KEY
```

No architectural changes. No code restructuring. The clause-driven pipeline, agent definitions, KB retrieval, eval layer, and API are all production-ready as written.

---

## Knowledge Base

12 curated entries covering the highest-risk clause categories in commercial contracts:

| Category | Content |
|----------|---------|
| Clause Analysis | Limitation of liability, indemnification, termination, confidentiality, IP, force majeure, dispute resolution, payment terms, warranties |
| Legal Principles | Common Law fundamentals, contra proferentem, variation clauses, waiver, severability |
| International Standards | UNIDROIT Principles — good faith, hardship, gross disparity, interpretation |
| Contract Type Guidance | Employment contracts, NDAs, service agreements |
| Risk Framework | CRITICAL / HIGH / MEDIUM / LOW assessment methodology |

All content is grounded in publicly available Common Law principles and UNIDROIT standards — applicable across 80+ jurisdictions including UK, USA, Canada, Australia, Nigeria, Kenya, Singapore, and India.

Precision over volume. Twelve well-structured entries covering the nine most common high-risk clause types outperform hundreds of scraped paragraphs for this use case. The retrieval layer returns the two most relevant entries per query — sufficient context without context window overflow.

---

## Evaluation

LLM-as-judge scoring on three weighted axes. A separate model handles evaluation — keeping the judge independent from the agents being evaluated.

| Axis | Weight | Measures |
|------|--------|---------|
| Clause Detection | 40% | Did the system identify all significant clauses present? |
| Legal Grounding | 35% | Are findings cited against specific legal standards? |
| Recommendation Specificity | 25% | Are recommendations specific and immediately actionable? |

Pass threshold: ≥ 0.70

**On free tier**, eval scores reflect the truncated analysis — scores for clause detection will be lower than production because the model covers fewer clauses. Legal grounding and recommendation specificity scores are more reliable as indicators since they measure quality of what was produced, not coverage.

Evaluation results are logged to `data/eval_log.jsonl` and accessible via `/eval/history` and `/eval/summary`.

---

## Clause Types Analysed

| Type | What It Covers |
|------|---------------|
| Limitation of Liability | Caps on financial exposure, exclusions, carve-outs |
| Indemnification | Third-party claims, broad indemnity red flags |
| Termination | Notice periods, cure rights, convenience termination |
| Confidentiality | Scope, duration, standard exclusions |
| Intellectual Property | Background IP, assignment scope, work for hire |
| Force Majeure | Event scope, notice requirements, longstop dates |
| Dispute Resolution | Court vs arbitration, seat, asymmetric clauses |
| Payment Terms | Net days, late payment interest, set-off rights |
| Warranties | As-is disclaimers, warranty periods, IP non-infringement |
| Non-Compete | Scope, duration, geography — enforceability |
| Assignment | Consent requirements, change of control |
| Entire Agreement | Merger clauses, variation requirements |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ui` | GET | Web interface |
| `/analyze` | POST | Upload PDF — returns structured analysis |
| `/eval/history` | GET | Recent evaluation scores |
| `/eval/summary` | GET | Aggregate evaluation statistics |
| `/health` | GET | System health and readiness check |
| `/docs` | GET | Swagger API documentation |

### Example Response

```json
{
  "filename": "service_agreement.pdf",
  "contract_type": "Service Agreement",
  "overall_risk": "HIGH",
  "analysis_report": "CONTRACT ANALYSIS REPORT\n...",
  "processing_time": 94.2,
  "clauses_detected": 12,
  "pages": 8,
  "governing_law": "England and Wales",
  "eval": {
    "score": 0.741,
    "passed": true,
    "axis_scores": {
      "clause_detection": 0.65,
      "legal_grounding": 0.82,
      "recommendation_specificity": 0.78
    }
  }
}
```

Note: `clause_detection` score of 0.65 reflects free-tier partial coverage. `legal_grounding` and `recommendation_specificity` reflect quality of the clauses that were analysed.

---

## Local Setup

```bash
git clone https://github.com/victor-isuo/lexai.git
cd lexai
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key          # Four-agent analysis pipeline
GEMINI_API_KEY=your_key        # LLM-as-judge evaluation layer
LANGCHAIN_API_KEY=your_key     # LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=lexai
```

Run:
```bash
uvicorn main:app --reload --port 7860
```

Open `http://localhost:7860/ui` and upload any contract PDF.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | CrewAI 1.14.2 — sequential four-agent pipeline |
| LLM (Analysis) | Groq Llama-4-Scout 17B-16E — four-agent reasoning |
| LLM (Evaluation) | Groq Llama-3.3-70B — LLM-as-judge scoring |
| Retrieval | LangChain + ChromaDB + Sentence Transformers |
| Embeddings | all-MiniLM-L6-v2 |
| PDF Parsing | pdfplumber — structure-aware clause extraction |
| API | FastAPI 0.136.0 |
| Deployment | Hugging Face Spaces (Docker) |
| Observability | LangSmith tracing |

---

## Key Decisions

**Why CrewAI over LangGraph**

LangGraph provides fine-grained state machine control suited for systems with complex conditional branching and dynamic routing. Legal contract review is a linear specialist handoff — each agent has a defined role and passes its output to the next. CrewAI's role-based sequential process maps directly to how a law firm actually operates: a senior analyst extracts clauses, a researcher grounds each finding in established law, a risk specialist scores exposure, a plain language specialist writes actionable recommendations. Sequential is not a limitation here — it is the correct model for this workflow.

**Why clause-driven input over raw text slicing**

The previous approach passed `contract_text[:6000]` to the crew — a blind character slice that ignored document structure entirely. The current approach uses structured `ContractClause` objects from the parser: each clause carries its heading, full text, detected type, and pre-scanned risk signals. When the input cap is hit, the 12 highest-risk clauses are selected by signal priority — not the first 12 that happen to appear before character 6,000. This is architecturally correct even when the model's output quality is constrained by the underlying LLM tier.

**Why sequential over parallel**

Each agent's output is the next agent's input. The Risk Assessor cannot produce severity scores without the Legal Researcher's citations. The Plain Language Specialist cannot write recommendations before the Risk Assessor has justified each rating. Parallel execution would require merging incomplete, ungrounded outputs — and would not reduce total token consumption on a shared TPM budget.

**Why 12 knowledge base entries**

Precision over volume. Each entry is curated, citable, and grounded in publicly available legal standards. Twelve well-structured entries covering the nine most common high-risk clause types outperform hundreds of scraped paragraphs for this use case. The retrieval layer returns the two most relevant entries per query — sufficient grounding without context window pressure.

**Why LLM-as-judge evaluation**

Output quality in agentic systems cannot be measured by latency or error rate alone. A contract analysis report that runs without errors but misses the indemnity clause is a worse failure than one that errors and retries. The evaluation layer measures what matters: did the system find the clauses, are the findings legally grounded, are the recommendations specific enough to act on. Every response carries a score. This is the same evaluation-first approach used in AgentEval.

---

## Project Structure

```
lexai/
├── src/
│   ├── agents/
│   │   └── legal_crew.py          # CrewAI four-agent pipeline
│   ├── core/
│   │   ├── document_parser.py     # Structure-aware PDF clause extraction
│   │   └── knowledge_base.py      # ChromaDB legal knowledge base
│   ├── tools/
│   │   └── retrieval_tool.py      # KB search + risk signal detection tools
│   └── evaluation/
│       └── legal_eval.py          # LLM-as-judge evaluation layer
├── static/
│   └── index.html                 # Web interface
├── data/
│   ├── uploads/                   # Temporary upload storage (auto-cleaned)
│   └── eval_log.jsonl             # Evaluation history log
├── main.py                        # FastAPI application
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Relationship to Industrial AI Copilot

LexAI demonstrates domain transferability of the same agentic RAG architecture across high-stakes domains:

| System | Domain | Stakes | Agent Framework |
|--------|--------|--------|----------------|
| Industrial AI Copilot | Equipment fault diagnosis | Operational downtime, safety | LangGraph |
| LexAI | Contract risk analysis | Legal and financial liability | CrewAI |

Both systems use structure-aware ingestion before retrieval, curated knowledge bases over scraped corpora, and independent evaluation infrastructure that makes accuracy measurable rather than claimed. Both operate in domains where an incorrect AI output carries a real cost.

---

## Important Disclaimer

LexAI analysis is grounded in Common Law contract principles and internationally recognised standards. It is AI-assisted analysis for informational purposes only and **does not constitute legal advice**. For contracts with significant financial or operational consequences, engage a qualified legal practitioner in the relevant jurisdiction.

---

## Author

**Victor Isuo** — Applied LLM Systems Engineer

Building production-grade AI systems for high-stakes domains.

[GitHub](https://github.com/victor-isuo) · [LinkedIn](https://linkedin.com/in/victor-isuo-a02b65171) · [Industrial AI Copilot](https://victorisuo-industrial-ai-copilot.hf.space) · [AgentEval](https://victorisuo-agenteval.hf.space)
