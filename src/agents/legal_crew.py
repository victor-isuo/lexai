"""
Legal Crew — LexAI
====================
Four-agent CrewAI crew for contract analysis.

Why CrewAI not LangGraph:
Contract analysis maps naturally to sequential role handoff.
Same architecture decision every time — Analyst → Researcher →
Risk Assessor → Plain Language. LangGraph would add state complexity
that this workflow doesn't need.
"""

import os
import logging
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from src.tools.retrieval_tool import LegalKnowledgeBaseTool, ClauseRiskTool

logger = logging.getLogger(__name__)

def get_llm() -> LLM:
    """Initialise the Groq LLM for CrewAI."""
    return LLM(
        model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_retries=2,
    )

def build_crew(contract_text: str, contract_metadata: dict) -> Crew:
    """
    Build and return a configured CrewAI crew for contract analysis.
    Contract text is truncated to 6000 chars to keep tasks focused.
    """
    llm       = get_llm()
    kb_tool   = LegalKnowledgeBaseTool()
    risk_tool = ClauseRiskTool()

    contract_type = contract_metadata.get("contract_type", "Commercial Agreement")
    governing_law = (
        contract_metadata.get("governing_law") 
        or "Governing law not specified in this agreement")
    parties       = contract_metadata.get("parties", [])
    parties_str   = " and ".join(parties) if parties else "parties not identified in document"

    # Cap contract text — agents get focused context, not the whole document
    contract_excerpt = contract_text[:3500]

    # ── AGENT 1: Contract Analyst ─────────────────────────────────────────────
    contract_analyst = Agent(
        role="Senior Contract Analyst",
        goal=(
            "Extract and categorise every significant clause from the contract. "
            "Identify clause type, key obligation, and which party bears the burden."
        ),
        backstory=(
            "Senior contract analyst with 15 years reviewing commercial agreements "
            "across Common Law jurisdictions. You find the critical provisions buried "
            "in boilerplate that others miss."
        ),
        llm=llm,
        tools=[kb_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        max_retry_limit=1,
    )

    # ── AGENT 2: Legal Researcher ─────────────────────────────────────────────
    legal_researcher = Agent(
        role="Legal Research Specialist",
        goal=(
            "For each significant clause, retrieve applicable legal principles "
            "from the knowledge base. Ground every finding in established law. "
            "Identify what is standard versus what is missing."
        ),
        backstory=(
            "Legal research specialist with deep expertise in Common Law contract "
            "principles and UNIDROIT standards. You never make legal claims without "
            "citing an authoritative source."
        ),
        llm=llm,
        tools=[kb_tool, risk_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        max_retry_limit=1,
    )

    # ── AGENT 3: Risk Assessor ────────────────────────────────────────────────
    risk_assessor = Agent(
        role="Contract Risk Assessment Specialist",
        goal=(
            "Assess each clause for commercial and legal risk. "
            "Assign CRITICAL, HIGH, MEDIUM, or LOW with specific justification. "
            "Identify missing protections. Produce overall risk summary."
        ),
        backstory=(
            "Contract risk specialist who advises boards on contractual exposure. "
            "You cut through jargon to identify what actually matters commercially. "
            "Your assessments are precise, justified, and actionable."
        ),
        llm=llm,
        tools=[risk_tool, kb_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        max_retry_limit=1,
    )

    # ── AGENT 4: Plain Language Agent ────────────────────────────────────────
    plain_language_agent = Agent(
        role="Legal Plain Language Specialist",
        goal=(
            "Transform the technical analysis into a clear, structured report "
            "that a business person can understand and act on. "
            "Every recommendation must be specific and actionable."
        ),
        backstory=(
            "Legal communication specialist who bridges lawyers and business people. "
            "You believe a correct finding nobody acts on is worthless. "
            "Clear, precise, no unnecessary jargon."
        ),
        llm=llm,
        tools=[],
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        max_retry_limit=1,
    )

    # ── TASK 1: Clause Extraction ─────────────────────────────────────────────
    extraction_task = Task(
        description=f"""
Analyse this {contract_type} between {parties_str}.
Governing law: {governing_law}.

CONTRACT:
{contract_excerpt}

Extract all significant clauses. For each:
1. Clause type (e.g. Limitation of Liability, Termination, IP)
2. Key language quoted from the clause
3. Which party it favours
4. Any unusual or non-standard language

Use legal_knowledge_base_search to understand standard clause content.
Be exhaustive — list every significant clause you find.
""",
        expected_output=(
            "Structured list of all significant clauses with: type, quoted language, "
            "favoured party, and unusual provisions noted."
        ),
        agent=contract_analyst,
    )

    # ── TASK 2: Legal Research ────────────────────────────────────────────────
    research_task = Task(
        description=f"""
Using the extracted clauses, research each one against legal standards.

For each clause:
1. Search the knowledge base for applicable standards
2. Run the risk signal tool on concerning clause text
3. Note where this contract meets or falls short of standard protections
4. Cite your knowledge base source for each finding

Contract type: {contract_type} | Governing law: {governing_law}
""",
        expected_output=(
            "Legal research findings per clause with knowledge base citations. "
            "Where this contract meets standards and where it falls short."
        ),
        agent=legal_researcher,
        context=[extraction_task],
    )

    # ── TASK 3: Risk Assessment ───────────────────────────────────────────────
    risk_task = Task(
        description=f"""
Using the clause analysis and legal research, produce a risk assessment.

For each clause assign: CRITICAL / HIGH / MEDIUM / LOW with justification.
State the consequence if the clause triggers as written.

Then provide:
- Overall contract risk: CRITICAL / HIGH / MEDIUM / LOW
- Missing protections for a {contract_type}
- Top 3 priority issues before signing
- Which party this contract overall favours
""",
        expected_output=(
            "Risk level per clause with justification, overall risk, "
            "missing protections, top 3 priorities, balance assessment."
        ),
        agent=risk_assessor,
        context=[research_task],
    )

    # ── TASK 4: Final Report ──────────────────────────────────────────────────
    report_task = Task(
        description=f"""
Write the final contract analysis report in this exact format:

---
CONTRACT ANALYSIS REPORT
LexAI — AI Contract Intelligence

CONTRACT: {contract_type}
PARTIES: {parties_str}
GOVERNING LAW: {governing_law}
OVERALL RISK: [CRITICAL / HIGH / MEDIUM / LOW]

EXECUTIVE SUMMARY
[3-4 sentences: what this contract is, who it favours, key concerns]

KEY FINDINGS
[For each significant clause:]
CLAUSE: [Name]
RISK: [CRITICAL / HIGH / MEDIUM / LOW]
WHAT IT SAYS: [Plain English — one sentence]
WHY IT MATTERS: [The practical consequence — one sentence]
RECOMMENDATION: [Specific action to take]

MISSING PROTECTIONS
[List provisions absent but standard for this contract type]

PRIORITY ACTIONS BEFORE SIGNING
1. [Most urgent]
2. [Second most urgent]
3. [Third most urgent]

IMPORTANT DISCLAIMER
This analysis is grounded in Common Law contract principles and
internationally recognised standards. It is AI-assisted analysis
for informational purposes only and does not constitute legal advice.
For contracts with significant financial or operational consequences,
engage a qualified legal practitioner in the relevant jurisdiction.
---

Write clearly. Short sentences. No jargon unless essential.
All recommendations must be specific and actionable.
""",
        expected_output=(
            "Complete contract analysis report in the exact format specified. "
            "All sections present. Plain English. Specific actionable recommendations."
        ),
        agent=plain_language_agent,
        context=[risk_task],
    )

    crew = Crew(
        agents=[
            contract_analyst, legal_researcher,
            risk_assessor, plain_language_agent
        ],
        tasks=[extraction_task, research_task, risk_task, report_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew

def analyse_contract(
    contract_text: str,
    contract_metadata: dict,
    filename: str = "contract.pdf",
) -> dict:
    """Run the full contract analysis pipeline."""
    import time
    logger.info(f"Analysing: {filename}")
    start = time.time()

    try:
        crew   = build_crew(contract_text, contract_metadata)
        result = crew.kickoff()

        report_text  = result.raw if hasattr(result, 'raw') else str(result)
        overall_risk = "MEDIUM"
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if f"OVERALL RISK: {level}" in report_text:
                overall_risk = level
                break

        elapsed = round(time.time() - start, 2)
        logger.info(f"Analysis done in {elapsed}s — Risk: {overall_risk}")

        return {
            "analysis_report": report_text,
            "overall_risk":    overall_risk,
            "contract_type":   contract_metadata.get("contract_type", "Unknown"),
            "processing_time": elapsed,
            "filename":        filename,
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "analysis_report": f"Analysis failed: {str(e)}",
            "overall_risk":    "UNKNOWN",
            "contract_type":   "Unknown",
            "processing_time": round(time.time() - start, 2),
            "filename":        filename,
        }

