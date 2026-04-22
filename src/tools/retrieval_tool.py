"""
Retrieval Tool — LexAI
========================
Custom CrewAI tools that give agents access to the knowledge base
and clause risk analysis. Each agent calls these to ground their
analysis in legal principles rather than parametric LLM memory.
"""

import re
import logging
from dotenv import load_dotenv
load_dotenv()

from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

_vectorstore = None


def set_vectorstore(vs):
    global _vectorstore
    _vectorstore = vs


class LegalKnowledgeBaseTool(BaseTool):
    name:        str = "legal_knowledge_base_search"
    description: str = (
        "Search the LexAI legal knowledge base for contract law principles, "
        "clause analysis guidance, risk frameworks, and legal standards. "
        "Use this for every clause analysis to ground findings in established law. "
        "Input: a legal question or clause type as a plain string."
    )

    def _run(self, query: str) -> str:
        if _vectorstore is None:
            return "Knowledge base not initialised."
        try:
            from src.core.knowledge_base import query_knowledge_base
            results = query_knowledge_base(_vectorstore, query, k=2)
            if not results:
                return f"No relevant guidance found for: {query}"

            output = f"LEGAL KNOWLEDGE BASE — '{query}'\n{'='*50}\n\n"
            for i, doc in enumerate(results, 1):
                source     = doc.metadata.get("source", "Legal Standards")
                risk_level = doc.metadata.get("risk_level", "medium")
                output += f"[Source {i}] {source} | Risk Level: {risk_level}\n"
                output += "-" * 40 + "\n"
                output += doc.page_content[:500] + "\n\n"
            return output[:1000]
        except Exception as e:
            logger.error(f"KB search failed: {e}")
            return f"Search error: {str(e)}"


class ClauseRiskTool(BaseTool):
    name:        str = "analyse_clause_risk_signals"
    description: str = (
        "Scan a contract clause for automated risk signals. "
        "Input: the clause text as a plain string. "
        "Returns detected risk patterns with severity levels."
    )

    RISK_PATTERNS = {
        "unlimited_liability":     (r"unlimited\s+liabilit|no\s+cap|without\s+limit",         "CRITICAL"),
        "unilateral_modification": (r"may\s+modify\s+at\s+any\s+time|reserves\s+the\s+right\s+to\s+change", "HIGH"),
        "automatic_renewal":       (r"automatically\s+renew|auto.renew",                       "MEDIUM"),
        "broad_indemnity":         (r"any\s+and\s+all\s+claims|indemnif\w+\s+against\s+any",  "HIGH"),
        "one_sided_termination":   (r"may\s+terminat\w+\s+at\s+any\s+time",                   "HIGH"),
        "ip_broad_assignment":     (r"all\s+intellectual\s+property|assign\s+all\s+right",     "HIGH"),
        "no_cure_period":          (r"terminat\w+\s+immediately\s+upon\s+breach",              "MEDIUM"),
        "waiver_of_jury":          (r"waive\w*\s+jury\s+trial",                                "MEDIUM"),
        "liquidated_damages":      (r"liquidated\s+damages|penalty\s+of\s+\$",                 "MEDIUM"),
        "perpetual_obligation":    (r"in\s+perpetuity|perpetual\s+licen",                      "LOW"),
    }

    def _run(self, clause_text: str) -> str:
        found = []
        for name, (pattern, severity) in self.RISK_PATTERNS.items():
            if re.search(pattern, clause_text, re.IGNORECASE):
                found.append((name, severity))

        if not found:
            return "No automated risk signals detected in this clause."

        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        found.sort(key=lambda x: order.get(x[1], 4))

        output = f"RISK SIGNALS DETECTED: {len(found)}\n\n"
        for name, severity in found:
            output += f"  [{severity}] {name.replace('_', ' ').title()}\n"
        output += f"\nHighest Severity: {found[0][1]}"
        return output
