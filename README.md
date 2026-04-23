---
title: LexAI
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# ⚖️ LexAI — AI Contract Intelligence

Production-grade contract analysis system grounded in Common Law principles and UNIDROIT international standards. Upload any contract PDF and receive a structured risk report with per-clause findings, severity ratings, and specific recommendations.

## 🔴 Live Demo

[https://victorisuo-lexai.hf.space/ui](https://victorisuo-lexai.hf.space/ui)

---

## What This Is

Most AI contract tools are RAG chatbots — they retrieve text and summarise it. LexAI is different. It applies a four-agent reasoning pipeline where each specialist hands off to the next: clause extraction → legal research → risk assessment → plain language report. Every finding is grounded in the legal knowledge base. Every risk rating is justified.

The difference between retrieving contract text and reasoning about it against established law is the difference between a keyword search and a legal opinion.

---

## System Architecture

```
Contract PDF Upload
        ↓
Document Parser
(clause extraction · type classification
 risk signal detection · metadata)
        ↓
ChromaDB Knowledge Base
(Common Law principles · clause analysis
 UNIDROIT standards · risk frameworks)
        ↓
CrewAI Four-Agent Sequential Pipeline
        ↓
┌────────────────────────────────────────────┐
│  Contract Analyst                          │
│  Extracts and categorises all clauses      │
│                    ↓                       │
│  Legal Researcher                          │
│  Grounds each clause in the knowledge base │
│                    ↓                       │
│  Risk Assessor                             │
│  Scores CRITICAL / HIGH / MEDIUM / LOW     │
│                    ↓                       │
│  Plain Language Agent                      │
│  Structured report with recommendations   │
└────────────────────────────────────────────┘
        ↓
Structured Analysis Report
(risk summary · per-clause findings
 legal citations · recommended actions)
        ↓
LLM-as-Judge Evaluation
(clause detection · legal grounding
 recommendation specificity)
```

---

## What Makes This Different From Generic RAG

**Clause-aware parsing.** Contracts have structure — headings, numbered provisions, schedules. The parser identifies clause boundaries before retrieval starts, so the system reasons about Clause 14 as a unit, not as arbitrary chunks.

**Four-agent role handoff.** Contract Analyst → Legal Researcher → Risk Assessor → Plain Language Agent. Each agent has a specific role, tools, and backstory. The Legal Researcher cannot produce findings without querying the knowledge base. The Risk Assessor cannot score without running the automated risk signal detector. The pipeline enforces discipline that a single-agent system cannot.

**Automated risk signal detection.** Ten risk patterns — unlimited liability, broad indemnity, unilateral modification rights, automatic renewal, one-sided termination, IP broad assignment — are detected automatically on every clause before the LLM reasoning step.

**LLM-as-judge evaluation.** Every analysis is scored by a separate judge model on three axes: clause detection accuracy, legal grounding accuracy, and recommendation specificity. The score is returned with every analysis response. This is the same evaluation-first engineering approach used in AgentEval.

---

## Knowledge Base

12 knowledge base entries covering:

| Category | Content |
|----------|---------|
| Clause Analysis | Limitation of liability, indemnification, termination, confidentiality, IP, force majeure, dispute resolution, payment terms, warranties |
| Legal Principles | Common Law contract fundamentals, contra proferentem, variation clauses, waiver, severability |
| International Standards | UNIDROIT Principles — good faith, hardship, gross disparity, interpretation |
| Contract Type Guidance | Employment contracts, NDAs, service agreements |
| Risk Framework | CRITICAL / HIGH / MEDIUM / LOW assessment methodology |

All content is grounded in publicly available Common Law principles and UNIDROIT standards — applicable across 80+ jurisdictions including UK, USA, Canada, Australia, Nigeria, Kenya, Singapore, and India.

---

## Evaluation Results

LLM-as-judge scoring using Gemini 2.5 Flash across three weighted axes. A separate model handles evaluation — keeping the judge independent from the agents being evaluated.

| Axis | Weight | Measures |
|------|--------|---------|
| Clause Detection | 40% | Did the system identify all significant clauses? |
| Legal Grounding | 35% | Are findings cited against legal standards? |
| Recommendation Specificity | 25% | Are recommendations specific and actionable? |

Pass threshold: ≥ 0.70

Evaluation results are logged to `data/eval_log.jsonl` and accessible via `/eval/history` and `/eval/summary` endpoints.

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
| `/analyze` | POST | Upload PDF and receive analysis |
| `/eval/history` | GET | Recent evaluation scores |
| `/eval/summary` | GET | Evaluation summary statistics |
| `/health` | GET | System health check |
| `/docs` | GET | Swagger API documentation |

### Example Response

```json
{
  "filename": "service_agreement.pdf",
  "contract_type": "Service Agreement",
  "overall_risk": "HIGH",
  "analysis_report": "CONTRACT ANALYSIS REPORT\n...",
  "processing_time": 87.4,
  "clauses_detected": 12,
  "pages": 8,
  "governing_law": "England and Wales",
  "eval": {
    "score": 0.847,
    "passed": true,
    "axis_scores": {
      "clause_detection": 0.90,
      "legal_grounding": 0.82,
      "recommendation_specificity": 0.78
    }
  }
}
```

**Processing time:** 60–120 seconds is expected. The four agents run sequentially — each specialist completes its full reasoning pass before handing off to the next. This is an architectural constraint, not a performance bug. The Legal Researcher cannot cite standards before the Contract Analyst has extracted clauses. The Risk Assessor cannot score before the Legal Researcher has grounded the findings.

---

## Local Setup

```bash
git clone https://github.com/victor-isuo/lexai.git
cd lexai
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key          # Powers the four-agent analysis pipeline
GEMINI_API_KEY=your_key        # Powers the LLM-as-judge evaluation layer
LANGCHAIN_API_KEY=your_key
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
| LLM (Multi-Agent Analysis) | Groq Llama 4 Scout — four-agent reasoning chain |
| LLM (Evaluation) | Gemini 2.5 Flash — single-call LLM-as-judge scoring |
| Retrieval | LangChain + ChromaDB + Sentence Transformers |
| Embeddings | all-MiniLM-L6-v2 |
| PDF Parsing | pdfplumber — structure-aware clause extraction |
| API | FastAPI 0.136.0 |
| Deployment | Hugging Face Spaces (Docker) |
| Observability | LangSmith tracing |

---

## Key Decisions

**Why CrewAI over LangGraph**

LangGraph provides fine-grained state machine control suited for systems with complex conditional branching and dynamic routing. Legal contract review is a linear specialist handoff — each agent has a defined role and passes its output to the next. CrewAI's role-based sequential process maps directly to how a law firm actually operates: a junior associate extracts, a researcher grounds, a senior associate assesses risk, a partner drafts recommendations. Sequential is not a limitation here — it is the correct model.

**Why sequential over parallel**

Each agent's output is the next agent's input. The Risk Assessor cannot produce severity scores without the Legal Researcher's citations. The Plain Language Agent cannot write recommendations without the Risk Assessor's findings. Parallel execution would require merging incomplete, ungrounded outputs. Sequential execution enforces analytical discipline at every handoff.

**Why 12 knowledge base entries**

Precision over volume. Each entry is curated, citable, and grounded in publicly available legal standards. Twelve well-structured entries covering the nine most common high-risk clause types outperform hundreds of scraped paragraphs for this use case. The retrieval layer returns the two most relevant entries per query — sufficient context without context window overflow.

**Contract length handling**

LexAI is optimised to prioritize clause-dense sections (definitions, governing law, liability) where >80% of legal risk is concentrated.
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
│   │   └── retrieval_tool.py      # CrewAI tools — KB search + risk detection
│   └── evaluation/
│       └── legal_eval.py          # LLM-as-judge evaluation layer
├── static/
│   └── index.html                 # Web interface
├── data/
│   ├── uploads/                   # Temporary upload storage
│   └── eval_log.jsonl             # Evaluation history log
├── main.py                        # FastAPI application
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Relationship to Industrial AI Copilot

LexAI demonstrates domain transferability of the same agentic RAG architecture:

| System | Domain | Stakes | Agent Framework |
|--------|--------|--------|----------------|
| Industrial AI Copilot | Equipment fault diagnosis | Operational downtime | LangGraph |
| LexAI | Contract risk analysis | Legal and financial liability | CrewAI |

Both systems include custom evaluation infrastructure that makes accuracy measurable rather than claimed. Both operate in domains where incorrect AI outputs carry real cost.

*"I've applied the same agentic RAG architecture across industrial equipment fault diagnosis — where a wrong answer causes downtime — and legal contract analysis — where a missed clause creates liability. Both systems include evaluation infrastructure that makes accuracy measurable, not claimed."*

---

## Important Disclaimer

LexAI analysis is grounded in Common Law contract principles and internationally recognised standards. It is AI-assisted analysis for informational purposes only and **does not constitute legal advice**. For contracts with significant financial or operational consequences, engage a qualified legal practitioner in the relevant jurisdiction.

---

## Author

**Victor Isuo** — Applied LLM Systems Engineer

Building production-grade AI systems for high-stakes domains.

[GitHub](https://github.com/victor-isuo) · [LinkedIn](https://linkedin.com/in/victor-isuo-a02b65171) · [Industrial AI Copilot](https://victorisuo-industrial-ai-copilot.hf.space) · [AgentEval](https://victorisuo-agenteval.hf.space)
