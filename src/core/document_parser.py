"""
Document Parser — LexAI
========================
Parses contract PDFs into structured data that the agent system
can reason over.

Why this matters:
A naive RAG system chunks a contract by character count and hopes
the retriever finds the right chunk. That fails for legal documents
because a clause can span multiple paragraphs, and the context of
"Clause 14.2" is meaningless without knowing what Clause 14 says.

This parser does it properly:
- Extracts the full text with page positions
- Identifies clause boundaries using structural patterns
- Detects contract metadata: parties, dates, governing law
- Classifies clauses by type before retrieval even starts
- Preserves the original clause text for citation

This is the same principle as the Industrial AI Copilot —
structure-aware ingestion produces better downstream reasoning.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ContractClause:
    """A single identified clause from a contract."""
    clause_id:    str
    clause_type:  str        # indemnity, liability, termination, etc.
    heading:      str
    text:         str
    page_num:     int
    char_start:   int
    risk_signals: list[str]  # keywords that flag potential risk


@dataclass
class ContractMetadata:
    """Extracted metadata from the contract."""
    contract_type:    str   = "Unknown"
    parties:          list  = field(default_factory=list)
    effective_date:   str   = ""
    governing_law:    str   = ""
    jurisdiction:     str   = ""
    total_pages:      int   = 0
    raw_text_length:  int   = 0


@dataclass
class ParsedContract:
    """Complete parsed contract ready for agent analysis."""
    filename:       str
    metadata:       ContractMetadata
    clauses:        list[ContractClause]
    full_text:      str
    raw_pages:      list[dict]   # [{page_num, text}]
    parse_warnings: list[str]


# ── Clause Type Patterns ──────────────────────────────────────────────────────

CLAUSE_TYPE_PATTERNS = {
    "limitation_of_liability": [
        r"limit\w*\s+of\s+liabilit",
        r"liabilit\w*\s+shall\s+not\s+exceed",
        r"maximum\s+liabilit",
        r"aggregate\s+liabilit",
        r"cap\s+on\s+liabilit",
    ],
    "indemnification": [
        r"indemnif",
        r"hold\s+harmless",
        r"defend\s+and\s+indemnif",
    ],
    "termination": [
        r"terminat\w+\s+of\s+agreement",
        r"terminat\w+\s+for\s+cause",
        r"terminat\w+\s+for\s+convenience",
        r"right\s+to\s+terminat",
        r"notice\s+of\s+terminat",
    ],
    "force_majeure": [
        r"force\s+majeure",
        r"act\s+of\s+god",
        r"circumstances\s+beyond\s+.{0,30}\s+control",
        r"unforeseeable\s+circumstances",
    ],
    "dispute_resolution": [
        r"arbitrat",
        r"dispute\s+resolution",
        r"mediat\w+",
        r"governing\s+law",
        r"jurisdiction",
    ],
    "confidentiality": [
        r"confidential",
        r"non.disclosure",
        r"proprietary\s+information",
        r"trade\s+secret",
    ],
    "intellectual_property": [
        r"intellectual\s+propert",
        r"ip\s+rights",
        r"copyright",
        r"patent",
        r"trademark",
        r"work\s+for\s+hire",
        r"assignment\s+of\s+rights",
    ],
    "payment_terms": [
        r"payment\s+terms",
        r"invoice\s+within",
        r"due\s+within\s+\d+\s+days",
        r"interest\s+on\s+late\s+payment",
        r"penalty\s+for\s+late",
    ],
    "warranties": [
        r"warrant\w+\s+and\s+represent",
        r"represent\w+\s+and\s+warrant",
        r"no\s+warrant",
        r"disclaim\w+\s+warrant",
    ],
    "non_compete": [
        r"non.compet",
        r"non.solicit",
        r"restraint\s+of\s+trade",
        r"covenant\s+not\s+to\s+compet",
    ],
    "assignment": [
        r"assignment\s+of\s+agreement",
        r"may\s+not\s+assign",
        r"shall\s+not\s+assign",
        r"transfer\w*\s+this\s+agreement",
    ],
    "entire_agreement": [
        r"entire\s+agreement",
        r"supersedes\s+all\s+prior",
        r"merger\s+clause",
        r"integration\s+clause",
    ],
}

# Signals that indicate a clause carries elevated risk
RISK_SIGNAL_PATTERNS = {
    "unlimited_liability":     r"unlimited\s+liabilit|no\s+cap|without\s+limit",
    "unilateral_modification": r"may\s+modify\s+at\s+any\s+time|reserves\s+the\s+right\s+to\s+change",
    "automatic_renewal":       r"automatically\s+renew|auto.renew",
    "broad_indemnity":         r"any\s+and\s+all\s+claims|indemnif\w+\s+against\s+any",
    "one_sided_termination":   r"may\s+terminat\w+\s+at\s+any\s+time|immediate\s+terminat",
    "governing_law_mismatch":  r"laws\s+of\s+(delaware|california|new\s+york|england)",
    "waiver_of_jury":          r"waive\w*\s+jury\s+trial|waiver\s+of\s+jury",
    "liquidated_damages":      r"liquidated\s+damages|penalty\s+of",
    "ip_assignment_broad":     r"all\s+intellectual\s+property|assign\s+all\s+right",
    "exclusivity":             r"exclusive\w*\s+provider|shall\s+not\s+engage\s+other",
}

# Clause heading patterns to detect section boundaries
HEADING_PATTERNS = [
    r"^(\d+)\.\s+([A-Z][A-Z\s]{3,50})$",           # 1. LIMITATION OF LIABILITY
    r"^(\d+\.\d+)\s+([A-Z][A-Z\s]{3,50})$",         # 14.2 GOVERNING LAW
    r"^(Article\s+\w+)[:\s]+(.+)$",                  # Article XIV: Termination
    r"^(Section\s+\d+)[:\s]+(.+)$",                  # Section 5: Payment Terms
    r"^(CLAUSE\s+\d+)[:\s]+(.+)$",                   # CLAUSE 3: Indemnification
    r"^\[?([A-Z][A-Z\s]{4,40})\]?$",                 # CONFIDENTIALITY
]


# ── Core Parser ───────────────────────────────────────────────────────────────

class ContractParser:
    """
    Parses contract PDFs into structured ContractClause objects.

    The parser:
    1. Extracts raw text preserving page positions
    2. Detects clause boundaries from heading patterns
    3. Classifies each clause by type
    4. Flags risk signals in each clause
    5. Extracts top-level metadata (parties, governing law)
    """

    def parse(self, pdf_path: str) -> ParsedContract:
        """
        Parse a contract PDF into structured data.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ParsedContract with metadata, clauses, and full text
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Contract PDF not found: {pdf_path}")

        logger.info(f"Parsing contract: {path.name}")
        warnings = []

        # Extract raw text from each page
        raw_pages = self._extract_pages(str(path))
        if not raw_pages:
            warnings.append("No text could be extracted from PDF")
            return ParsedContract(
                filename=path.name,
                metadata=ContractMetadata(),
                clauses=[],
                full_text="",
                raw_pages=[],
                parse_warnings=warnings,
            )

        full_text = "\n\n".join(p["text"] for p in raw_pages)

        # Extract metadata
        metadata = self._extract_metadata(full_text, len(raw_pages))

        # Identify clauses
        clauses = self._identify_clauses(raw_pages, full_text)

        if not clauses:
            warnings.append(
                "No clause boundaries detected — full text will be used for analysis"
            )

        logger.info(
            f"Parsed {path.name}: {len(clauses)} clauses, "
            f"{len(raw_pages)} pages, type={metadata.contract_type}"
        )

        return ParsedContract(
            filename=path.name,
            metadata=metadata,
            clauses=clauses,
            full_text=full_text,
            raw_pages=raw_pages,
            parse_warnings=warnings,
        )

    def _extract_pages(self, pdf_path: str) -> list[dict]:
        """Extract text from each page using pdfplumber."""
        pages = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    text = self._clean_text(text)
                    if text.strip():
                        pages.append({"page_num": i + 1, "text": text})
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
        return pages

    def _clean_text(self, text: str) -> str:
        """Remove common PDF artefacts."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # Remove page numbers standing alone
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text.strip()

    def _extract_metadata(self, text: str, total_pages: int) -> ContractMetadata:
        """Extract contract metadata from the full text."""
        meta = ContractMetadata(
            total_pages=total_pages,
            raw_text_length=len(text),
        )

        # Contract type detection
        text_lower = text.lower()
        type_signals = {
            "Employment Agreement":      ["employment agreement", "contract of employment", "offer of employment"],
            "Non-Disclosure Agreement":  ["non-disclosure", "nda", "confidentiality agreement"],
            "Service Agreement":         ["service agreement", "services agreement", "consulting agreement"],
            "Software License":          ["software license", "licence agreement", "end user license"],
            "Partnership Agreement":     ["partnership agreement", "joint venture"],
            "Lease Agreement":           ["lease agreement", "tenancy agreement", "rental agreement"],
            "Sale Agreement":            ["sale agreement", "purchase agreement", "sale of goods"],
            "Shareholder Agreement":     ["shareholder agreement", "shareholders agreement"],
            "Loan Agreement":            ["loan agreement", "credit agreement", "facility agreement"],
        }
        for contract_type, signals in type_signals.items():
            if any(s in text_lower for s in signals):
                meta.contract_type = contract_type
                break

        # Governing law
        gov_law_patterns = [
            r"governed\s+by\s+the\s+laws?\s+of\s+([A-Za-z\s,]+?)(?:\.|,|\n)",
            r"governing\s+law[:\s]+([A-Za-z\s,]+?)(?:\.|,|\n)",
            r"laws?\s+of\s+(England|Nigeria|New\s+York|California|Delaware|Scotland)",
        ]
        for pattern in gov_law_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                meta.governing_law = match.group(1).strip()[:80]
                break

        # Parties — look for "between X and Y" patterns near the top
        party_patterns = [
            r"between\s+([A-Z][A-Za-z\s,\.]+?)\s+(?:\(["']?(?:Company|Employer|Client|Licensor|Party\s+A)['"]\))\s+and\s+([A-Z][A-Za-z\s,\.]+?)\s+(?:\(["']?(?:Employee|Contractor|Service\s+Provider|Licensee|Party\s+B)['"]\))",
            r"THIS\s+AGREEMENT\s+is\s+(?:made\s+)?(?:and\s+entered\s+into\s+)?(?:by\s+and\s+)?between\s+(.+?)\s+and\s+(.+?)(?:,|\.|collectively)",
        ]
        for pattern in party_patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE | re.DOTALL)
            if match:
                meta.parties = [match.group(1).strip()[:100], match.group(2).strip()[:100]]
                break

        # Effective date
        date_patterns = [
            r"effective\s+(?:as\s+of\s+)?(?:date[:\s]+)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
            r"dated\s+(?:this\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})",
            r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                meta.effective_date = match.group(1).strip()
                break

        return meta

    def _identify_clauses(
        self, raw_pages: list[dict], full_text: str
    ) -> list[ContractClause]:
        """
        Identify clause boundaries and extract structured clauses.
        Uses heading pattern detection to split the contract.
        """
        clauses = []
        combined_lines = []

        # Build a line index with page references
        for page in raw_pages:
            for line in page["text"].split("\n"):
                combined_lines.append({
                    "text": line,
                    "page_num": page["page_num"],
                })

        # Find heading positions
        heading_positions = []
        for i, line_data in enumerate(combined_lines):
            line = line_data["text"].strip()
            if not line:
                continue
            for pattern in HEADING_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    heading_positions.append({
                        "line_idx": i,
                        "heading":  line,
                        "page_num": line_data["page_num"],
                    })
                    break

        # Extract text between headings
        for j, pos in enumerate(heading_positions):
            start_idx  = pos["line_idx"] + 1
            end_idx    = (
                heading_positions[j + 1]["line_idx"]
                if j + 1 < len(heading_positions)
                else len(combined_lines)
            )

            clause_lines = [
                combined_lines[k]["text"]
                for k in range(start_idx, min(end_idx, start_idx + 200))
                if combined_lines[k]["text"].strip()
            ]
            clause_text = " ".join(clause_lines).strip()

            if len(clause_text) < 20:
                continue

            clause_type   = self._classify_clause(pos["heading"], clause_text)
            risk_signals  = self._detect_risk_signals(clause_text)
            clause_id     = f"clause_{j+1:03d}"

            clauses.append(ContractClause(
                clause_id=clause_id,
                clause_type=clause_type,
                heading=pos["heading"],
                text=clause_text[:2000],  # cap for embedding
                page_num=pos["page_num"],
                char_start=j,
                risk_signals=risk_signals,
            ))

        # If no headings found, fall back to paragraph chunking
        if not clauses:
            clauses = self._chunk_by_paragraph(raw_pages)

        return clauses

    def _classify_clause(self, heading: str, text: str) -> str:
        """Classify a clause into a known type using pattern matching."""
        combined = (heading + " " + text).lower()
        for clause_type, patterns in CLAUSE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return clause_type
        return "general"

    def _detect_risk_signals(self, text: str) -> list[str]:
        """Detect risk indicator patterns in clause text."""
        signals = []
        for signal_name, pattern in RISK_SIGNAL_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                signals.append(signal_name)
        return signals

    def _chunk_by_paragraph(self, raw_pages: list[dict]) -> list[ContractClause]:
        """
        Fallback: chunk by paragraph when no headings detected.
        Used for poorly-formatted or scanned contracts.
        """
        clauses = []
        clause_id = 0
        for page in raw_pages:
            paragraphs = [p.strip() for p in page["text"].split("\n\n") if len(p.strip()) > 80]
            for para in paragraphs:
                clause_id += 1
                risk_signals = self._detect_risk_signals(para)
                clauses.append(ContractClause(
                    clause_id=f"para_{clause_id:03d}",
                    clause_type=self._classify_clause("", para),
                    heading=f"Paragraph {clause_id}",
                    text=para[:2000],
                    page_num=page["page_num"],
                    char_start=clause_id,
                    risk_signals=risk_signals,
                ))
        return clauses
