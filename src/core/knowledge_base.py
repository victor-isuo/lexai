"""
Knowledge Base — LexAI
=======================
ChromaDB vector store containing contract law principles,
clause interpretations, and risk guidance.

This is the retrieval layer that gives the agents legal grounding.
Without this, the agents would reason from parametric LLM memory —
which is unreliable and uncitable.

With this, every agent finding can be traced to a specific
knowledge base entry.

Content categories:
- Common Law contract principles
- Standard clause analysis (what each clause type means and risks)
- UNIDROIT Principles of International Commercial Contracts
- Negotiation guidance per clause type
- Red flags and protective language examples
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CHROMA_PATH      = Path("data/lexai_kb")
COLLECTION_NAME  = "lexai_knowledge"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"


def get_embeddings():
    """Get embedding model — same as Industrial AI Copilot for consistency."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def build_knowledge_base() -> Chroma:
    """
    Build the LexAI knowledge base from the built-in legal content.
    Creates ChromaDB collection if it doesn't exist.
    """
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )

    # Check if already populated
    existing = vectorstore.get()
    if existing and len(existing.get("ids", [])) > 0:
        logger.info(f"Knowledge base loaded: {len(existing['ids'])} documents")
        return vectorstore

    # Build from content
    logger.info("Building knowledge base from legal content...")
    docs = _build_documents()
    vectorstore.add_documents(docs)
    logger.info(f"Knowledge base built: {len(docs)} documents indexed")

    return vectorstore


def load_knowledge_base() -> Chroma:
    """Load existing knowledge base or build if not present."""
    return build_knowledge_base()


def query_knowledge_base(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
    clause_type: str = None,
) -> list[Document]:
    """
    Query the knowledge base with optional clause type filtering.

    Args:
        vectorstore:  The ChromaDB vector store
        query:        Natural language query
        k:            Number of results to return
        clause_type:  Optional filter by clause type
    """
    if clause_type:
        results = vectorstore.similarity_search(
            query, k=k,
            filter={"clause_type": clause_type}
        )
        if not results:
            results = vectorstore.similarity_search(query, k=k)
    else:
        results = vectorstore.similarity_search(query, k=k)
    return results


# ── Knowledge Base Content ────────────────────────────────────────────────────

def _build_documents() -> list[Document]:
    """Build the knowledge base documents from legal content."""
    docs = []
    for entry in _get_legal_content():
        docs.append(Document(
            page_content=entry["content"],
            metadata={
                "source":      entry["source"],
                "clause_type": entry["clause_type"],
                "category":    entry["category"],
                "risk_level":  entry.get("risk_level", "medium"),
            }
        ))
    return docs


def _get_legal_content() -> list[dict]:
    """
    Core legal knowledge base content.
    Grounded in Common Law principles and internationally
    recognised contract standards (UNIDROIT).
    """
    return [

        # ── LIMITATION OF LIABILITY ──────────────────────────────────────────
        {
            "clause_type": "limitation_of_liability",
            "category":    "clause_analysis",
            "source":      "Common Law Contract Principles",
            "risk_level":  "high",
            "content": """LIMITATION OF LIABILITY CLAUSES — Analysis and Risk Assessment

A limitation of liability clause caps the maximum financial exposure of one or both parties in the event of breach or loss. Under Common Law, such clauses are generally enforceable provided they are: (1) clearly drafted, (2) reasonably brought to the other party's attention, and (3) not unconscionable.

STANDARD FORMS:
- Cap tied to contract value: "Liability shall not exceed the total fees paid in the 12 months preceding the claim." This is the most common and generally balanced form.
- Cap tied to insurance: "Liability shall not exceed the limits of the party's professional indemnity insurance." Acceptable if insurance minimums are specified.
- Mutual cap: Both parties' liability is equally capped. Generally fair.
- One-sided cap: Only one party's liability is limited. Favours the drafter — examine carefully.

RED FLAGS:
1. No cap at all on one party while the other is capped — creates unlimited exposure.
2. Exclusion of consequential damages that includes "loss of profits" — can eliminate the most significant heads of loss.
3. Cap set below the likely value of a claim — makes the clause commercially meaningless as protection.
4. Carve-outs that are too narrow — death, personal injury, and fraud should always be excluded from liability caps under Common Law.

PROTECTIVE LANGUAGE TO LOOK FOR:
- "Notwithstanding the above, nothing in this agreement shall limit liability for death or personal injury caused by negligence, fraud or fraudulent misrepresentation."
- Specific carve-outs for IP infringement and data protection breaches.

NEGOTIATION GUIDANCE:
If you are the party receiving services: push for the highest possible cap (at minimum, 12 months' fees). Ensure consequential loss exclusions do not eliminate your ability to claim loss of profits from a material breach. Always insist on carve-outs for fraud and wilful misconduct.""",
        },

        # ── INDEMNIFICATION ──────────────────────────────────────────────────
        {
            "clause_type": "indemnification",
            "category":    "clause_analysis",
            "source":      "Common Law Contract Principles",
            "risk_level":  "high",
            "content": """INDEMNIFICATION CLAUSES — Analysis and Risk Assessment

An indemnification clause requires one party to compensate the other for specified losses, claims, or expenses — including those arising from third-party claims. Unlike a damages claim for breach, indemnities can be triggered without fault and can cover losses that go beyond the normal measure of contract damages.

TYPES OF INDEMNITY:
1. Third-party indemnity: Party A indemnifies Party B against claims from third parties arising from A's acts or omissions. Standard and generally acceptable.
2. Mutual indemnity: Both parties indemnify each other. Most balanced form.
3. Broad indemnity: "Any and all claims, losses, damages, costs and expenses of any nature whatsoever." This is the most dangerous form — it has virtually no limits.
4. IP indemnity: Indemnification specifically for intellectual property infringement claims. Common in technology contracts — the licensor should always provide this.

RED FLAGS:
1. "Any and all claims" language without carve-outs for the indemnitee's own negligence — a party should not be indemnified for their own fault.
2. Indemnity that covers losses caused by the indemnitee — creates a moral hazard and is commercially unusual.
3. No cap on indemnity obligations — unlimited indemnities are one of the highest-risk provisions in any contract.
4. Consequential loss within indemnity scope — can create exposure many times the contract value.
5. Indemnity that survives termination indefinitely — check survival clause.

PROTECTIVE PROVISIONS:
- Limit indemnity to losses arising "directly and solely" from the indemnifying party's acts.
- Include a cap on indemnity obligations tied to the liability cap.
- Require the indemnified party to mitigate losses and give prompt notice of claims.
- Exclude coverage for the indemnitee's own negligence, fraud, or wilful misconduct.""",
        },

        # ── TERMINATION ─────────────────────────────────────────────────────
        {
            "clause_type": "termination",
            "category":    "clause_analysis",
            "source":      "Common Law Contract Principles",
            "risk_level":  "medium",
            "content": """TERMINATION CLAUSES — Analysis and Risk Assessment

Termination clauses govern when and how a party may bring the agreement to an end. Under Common Law, termination rights must be clearly expressed — if a right to terminate is not explicitly granted, courts will imply a right only in limited circumstances (repudiatory breach, frustration).

TYPES OF TERMINATION:
1. Termination for cause / breach: The right to terminate if the other party materially breaches and fails to cure within a notice period. Most common and generally balanced.
2. Termination for convenience: The right to terminate without cause on notice. Heavily favours the party holding this right — examine carefully if it is one-sided.
3. Automatic termination: The agreement terminates automatically on specified events (insolvency, change of control). Generally acceptable.
4. Termination for regulatory change: Allows termination if regulatory requirements change. More common in financial services contracts.

RED FLAGS:
1. One-sided termination for convenience: If only one party can terminate for convenience, the other party has significant commercial risk — they may invest resources based on the contract and have it terminated with minimal notice.
2. Very short notice periods: A 7-day notice period for termination of a long-running services agreement may not give adequate time to transition.
3. No cure period: Termination on breach without an opportunity to cure is harsh — standard practice is 14-30 days' notice and opportunity to cure.
4. Termination on change of control without consent: Can be triggered by internal corporate restructuring.
5. Survival clauses that preserve all obligations post-termination — check which clauses survive and for how long.

NOTICE REQUIREMENTS:
Termination notices typically must be: (a) in writing, (b) served by a specified method (email, registered post), (c) served on a specified person. Failure to follow notice requirements can invalidate a purported termination.

CONSEQUENCES OF TERMINATION:
Check: (1) What accrued rights survive? (2) Are payments due for work completed before termination? (3) Are there wind-down obligations? (4) What happens to confidential information and intellectual property?""",
        },

        # ── CONFIDENTIALITY ──────────────────────────────────────────────────
        {
            "clause_type": "confidentiality",
            "category":    "clause_analysis",
            "source":      "Common Law Contract Principles",
            "risk_level":  "medium",
            "content": """CONFIDENTIALITY CLAUSES — Analysis and Risk Assessment

Confidentiality provisions restrict the use and disclosure of information exchanged under the agreement. In standalone NDAs they are the primary obligation; in service agreements they are typically ancillary. Under Common Law, a duty of confidence can arise even without an express clause if information is clearly confidential and the recipient knows it — but express provisions are always preferable for certainty.

DEFINITION OF CONFIDENTIAL INFORMATION:
The scope of what is "confidential" varies significantly:
- Broad definition: "All information disclosed by one party to the other, in any form." Includes publicly available information unless specifically excluded. Favours the disclosing party.
- Marked confidential: Only information marked "CONFIDENTIAL" at time of disclosure. Narrow and impractical for verbal disclosures.
- Standard definition: Information that is non-public and reasonably understood to be confidential, with standard exclusions. Most balanced approach.

STANDARD EXCLUSIONS (should always be present):
1. Information already in the public domain (not through breach).
2. Information already known to the recipient before disclosure.
3. Information independently developed by the recipient.
4. Information required to be disclosed by law or court order (with notice obligation).

RED FLAGS:
1. No standard exclusions — the confidentiality obligation becomes unreasonably broad.
2. Indefinite duration — perpetual confidentiality is commercially unusual except for trade secrets. 3-5 years is standard for general confidential information.
3. No exceptions for regulatory disclosure — can create compliance problems.
4. Obligation to keep the existence of the agreement confidential — can create difficulties with auditors, investors, or regulators.
5. Return/destruction obligations with no carve-out for backup copies maintained per retention policies.

DURATION:
- Trade secrets: Indefinite protection is appropriate.
- General confidential information: 2-5 years post-disclosure or post-termination.
- Personal data: Subject to data protection law requirements — not just contractual confidentiality.""",
        },

        # ── INTELLECTUAL PROPERTY ────────────────────────────────────────────
        {
            "clause_type": "intellectual_property",
            "category":    "clause_analysis",
            "source":      "Common Law Contract Principles",
            "risk_level":  "high",
            "content": """INTELLECTUAL PROPERTY CLAUSES — Analysis and Risk Assessment

IP provisions govern who owns intellectual property created under or in connection with the agreement. This is one of the highest-stakes areas of contract negotiation — IP ownership can determine the long-term commercial value of the relationship.

KEY CONCEPTS:
- Background IP: IP owned by a party before the agreement, or developed independently of it. Should remain with the original owner.
- Foreground IP: IP created during performance of the agreement. Ownership is the central negotiation point.
- Licence: Permission to use IP without transferring ownership.
- Assignment: Transfer of IP ownership — permanent and comprehensive.

OWNERSHIP MODELS:
1. Client owns all foreground IP: "All work product and deliverables shall be the exclusive property of the Client." Standard in bespoke development contracts. The service provider should ensure their background IP is explicitly excluded.
2. Service provider owns foreground IP, grants licence: Common in SaaS and product companies. The client gets a licence, not ownership. Check: is the licence broad enough for the client's needs?
3. Shared ownership: Joint ownership of foreground IP. Creates complications — in many Common Law jurisdictions, either joint owner can use IP without accounting to the other. Often inadvisable.

RED FLAGS:
1. "All intellectual property" assignment without background IP carve-out — the service provider may unintentionally assign tools and frameworks they rely on for other clients.
2. Assignment of IP that does not yet exist — "assigns all right, title and interest in all future IP" — is a very broad obligation.
3. Moral rights waiver — relevant in UK and Commonwealth jurisdictions where moral rights (right to attribution, integrity) exist.
4. No licence back to service provider for their own background IP incorporated into deliverables.
5. Work for hire language in jurisdictions where this doctrine does not apply automatically.

PROTECTIVE PROVISIONS:
- Explicit carve-out: "The above assignment excludes all Background IP, which shall remain the property of [Service Provider]."
- Definition of Background IP should be comprehensive — consider schedules listing specific tools, frameworks, and pre-existing works.""",
        },

        # ── FORCE MAJEURE ────────────────────────────────────────────────────
        {
            "clause_type": "force_majeure",
            "category":    "clause_analysis",
            "source":      "Common Law Contract Principles",
            "risk_level":  "medium",
            "content": """FORCE MAJEURE CLAUSES — Analysis and Risk Assessment

A force majeure clause excuses a party's non-performance when it is caused by specified extraordinary events beyond their control. Under Common Law (unlike Civil Law systems), there is no implied force majeure doctrine — if the contract does not contain such a clause, a party seeking relief for extraordinary events must rely on the narrow doctrine of frustration, which is difficult to establish.

SCOPE OF FORCE MAJEURE EVENTS:
Standard events: Acts of God, natural disasters, war, terrorism, strikes, pandemics, government actions, regulatory changes.
Technology-specific: Cyber attacks, infrastructure failures, internet outages.

REQUIREMENTS FOR FORCE MAJEURE RELIEF:
Under most Common Law contracts, a party claiming force majeure must establish:
1. The event falls within the defined force majeure events.
2. The event was beyond the party's reasonable control.
3. The event directly caused the failure to perform.
4. The party took reasonable steps to mitigate the effects.
5. Notice was given promptly (typically 5-14 days).

RED FLAGS:
1. No notice requirement — force majeure should require prompt written notice to the other party.
2. No mitigation obligation — parties should be required to use reasonable endeavours to overcome the force majeure event.
3. Indefinite suspension — force majeure should not permit indefinite non-performance; include a longstop date after which either party may terminate.
4. Economic hardship as force majeure — mere price increases or market changes are not generally accepted as force majeure in Common Law jurisdictions.
5. One-sided application — force majeure should apply to both parties.

POST COVID-19 CONSIDERATIONS:
Courts in multiple Common Law jurisdictions have clarified that pandemic-related disruption does not automatically constitute force majeure — the clause must specifically reference pandemics, governmental restrictions, or public health emergencies.""",
        },

        # ── DISPUTE RESOLUTION ───────────────────────────────────────────────
        {
            "clause_type": "dispute_resolution",
            "category":    "clause_analysis",
            "source":      "Common Law Contract Principles",
            "risk_level":  "medium",
            "content": """DISPUTE RESOLUTION CLAUSES — Analysis and Risk Assessment

Dispute resolution clauses determine how disagreements will be resolved — through courts, arbitration, mediation, or a combination. This is a critical clause that is frequently overlooked until a dispute actually arises.

OPTIONS:
1. Court litigation: Disputes resolved through the national court system. Public, potentially slower, but established appellate process.
2. Arbitration: Private dispute resolution before one or more arbitrators. Awards are generally binding and enforceable internationally under the New York Convention (160+ countries).
3. Mediation: Non-binding facilitated negotiation. Typically included as a required step before arbitration or litigation.
4. Tiered clause: Negotiation → Mediation → Arbitration/Litigation. Most sophisticated approach — encourages resolution at the lowest cost level.
5. Expert determination: Used for specific technical or financial disputes. Expert's decision may be binding or non-binding.

ARBITRATION CONSIDERATIONS:
- Seat of arbitration determines the procedural law. Choose carefully — English-seat arbitration benefits from well-developed supervisory courts.
- Institutional vs ad hoc: ICC, LCIA, SIAC rules provide structure and administration. Ad hoc arbitration under UNCITRAL Rules is cheaper for smaller disputes.
- Number of arbitrators: Sole arbitrator is faster and cheaper; three-person tribunal is more appropriate for high-value disputes.
- Language of arbitration: Should be specified explicitly.

RED FLAGS:
1. Jurisdiction clause pointing to a forum that is inconvenient, expensive, or legally uncertain for your operations.
2. No mediation step — going straight to arbitration or litigation is expensive and relationship-damaging.
3. Asymmetric clause: One party can choose either courts or arbitration while the other is limited to one option. Heavily favours the party with the choice.
4. Arbitration seat in a jurisdiction that is not party to the New York Convention — makes award enforcement internationally very difficult.
5. Confidentiality of arbitration proceedings — ensure this is explicit if required, as it is not automatic under all institutional rules.""",
        },

        # ── PAYMENT TERMS ────────────────────────────────────────────────────
        {
            "clause_type": "payment_terms",
            "category":    "clause_analysis",
            "source":      "Common Law Contract Principles",
            "risk_level":  "medium",
            "content": """PAYMENT TERMS — Analysis and Risk Assessment

Payment provisions govern when, how, and under what conditions payments must be made. Clear payment terms reduce disputes and protect both parties' cash flow.

STANDARD COMPONENTS:
1. Payment timing: Net 30 (30 days from invoice) is standard for B2B services. Net 60 or 90 favours the payer and creates cash flow risk for the recipient.
2. Invoice requirements: What information must an invoice contain to be valid? Missing requirements can delay payment legitimately.
3. Disputed invoices: How are disputes handled? A party should not be able to withhold payment of undisputed portions.
4. Late payment: Interest on late payment (typically Base Rate + 8% in UK; equivalent in other jurisdictions). Statutory rates apply in many Common Law jurisdictions.
5. Currency: Specified currency of payment and FX risk allocation for cross-border contracts.
6. Set-off rights: The right to deduct amounts owed by the other party from payments due. Should be limited to undisputed, liquidated sums.

RED FLAGS:
1. Payment terms beyond Net 60 for services contracts — creates significant cash flow risk and is commercially unusual.
2. "Payment subject to client satisfaction" without objective criteria — creates unlimited right to withhold payment.
3. No late payment interest — remove the incentive to pay on time.
4. Broad set-off rights — can be used to withhold legitimate payments on basis of disputed claims.
5. Payment in foreign currency without FX protection mechanism.
6. Milestone payments tied to subjective acceptance criteria without a deemed acceptance fallback — if no acceptance is given within X days, milestone is deemed accepted.

RETENTIONS:
In construction and professional services contracts, retention clauses withhold a percentage of payment until project completion. Ensure: (1) retention period is reasonable, (2) release mechanism is clear, (3) retention money is held in a separate account where large sums are involved.""",
        },

        # ── WARRANTIES ───────────────────────────────────────────────────────
        {
            "clause_type": "warranties",
            "category":    "clause_analysis",
            "source":      "Common Law Contract Principles",
            "risk_level":  "medium",
            "content": """WARRANTIES AND REPRESENTATIONS — Analysis and Risk Assessment

Warranties are contractual promises about the current or future state of facts. Representations are statements of fact that induce a party to enter the contract. Under Common Law, breach of a warranty gives rise to a claim for damages; a negligent or fraudulent misrepresentation can give rise to rescission of the contract.

TYPES:
1. Capacity warranties: "The party has full power and authority to enter this agreement." Standard and generally uncontroversial.
2. Title warranties: "The party has good and marketable title to the assets/IP being transferred." Critical in sale and IP assignment transactions.
3. Performance warranties: "The services will be performed with reasonable skill and care." Standard professional services warranty.
4. Product warranties: "The software/goods will conform to the specifications for X months." Standard product warranty.
5. No litigation warranty: "There are no pending or threatened claims that would materially affect performance." Important in acquisition and services contracts.

DISCLAIMER OF WARRANTIES:
"AS IS" and "WITHOUT WARRANTY" clauses attempt to exclude all implied warranties. Under Common Law:
- Implied terms (fitness for purpose, satisfactory quality under Sale of Goods Act) may be difficult to fully exclude in consumer contracts.
- In B2B contracts, broader exclusions are generally enforceable subject to reasonableness tests.
- Excluding the implied duty of reasonable skill and care in service contracts is generally subject to the test of reasonableness.

RED FLAGS:
1. Broad "as is" disclaimer in a services contract — may attempt to exclude the duty to perform with reasonable skill and care.
2. No warranty period specified — limits the window for making warranty claims.
3. Warranty remedy limited to "repair or replace" only — may be inadequate if defects are material.
4. No warranty for IP non-infringement — critical in technology contracts. If the deliverables infringe third-party IP, the client needs a remedy.
5. Survival of warranties post-termination not specified — check how long warranty claims can be brought.""",
        },

        # ── COMMON LAW PRINCIPLES ────────────────────────────────────────────
        {
            "clause_type": "general",
            "category":    "legal_principles",
            "source":      "Common Law Contract Principles",
            "risk_level":  "low",
            "content": """FUNDAMENTAL COMMON LAW CONTRACT PRINCIPLES

CONTRACT FORMATION:
A binding contract under Common Law requires: (1) offer, (2) acceptance, (3) consideration, (4) intention to create legal relations, and (5) capacity of the parties. A contract may be oral, written, or partly oral and partly written — though certain types (real property, guarantees) require writing to be enforceable.

CONTRA PROFERENTEM:
Ambiguous terms are construed against the party that drafted them. This principle provides protection against unclear drafting but should not be relied upon — clear drafting is always preferable.

ENTIRE AGREEMENT CLAUSES:
A clause stating that the written agreement constitutes the entire agreement between the parties and supersedes all prior representations. Prevents claims based on pre-contractual statements unless they were fraudulent.

VARIATION CLAUSES:
"No variation of this agreement shall be effective unless in writing and signed by the parties." Courts in Common Law jurisdictions have upheld such clauses, meaning informal modifications (email, verbal) may not be binding.

WAIVER:
A party may waive its rights under a contract — expressly or by conduct. Non-waiver clauses ("failure to exercise a right shall not constitute a waiver") are standard and important — without them, consistent non-enforcement of a right may be treated as a waiver.

SEVERABILITY:
If one provision is found to be void or unenforceable, a severability clause allows the remainder of the agreement to continue in force. Without it, a void clause could potentially invalidate the entire agreement.

GOVERNING LAW AND JURISDICTION:
Parties are generally free to choose the governing law of their contract (subject to mandatory law provisions). The choice affects which implied terms apply, how ambiguities are resolved, and what remedies are available.""",
        },

        # ── UNIDROIT PRINCIPLES ──────────────────────────────────────────────
        {
            "clause_type": "general",
            "category":    "legal_principles",
            "source":      "UNIDROIT Principles of International Commercial Contracts",
            "risk_level":  "low",
            "content": """UNIDROIT PRINCIPLES OF INTERNATIONAL COMMERCIAL CONTRACTS

The UNIDROIT Principles (2016 edition) provide internationally recognised rules for commercial contracts used across 160+ countries. They are widely referenced in international arbitration and serve as a model for contract drafting in cross-border transactions.

KEY PRINCIPLES RELEVANT TO CONTRACT ANALYSIS:

GOOD FAITH AND FAIR DEALING (Article 1.7):
Each party must act in accordance with good faith and fair dealing in international trade. Parties cannot exclude or limit this duty. Relevant when assessing whether a clause is being exercised in an oppressive manner.

DUTY TO NEGOTIATE IN GOOD FAITH (Article 2.1.15):
A party who negotiates in bad faith, including breaking off negotiations without justification when the other party has relied on continuation, is liable for losses caused.

HARDSHIP (Articles 6.2.1-6.2.3):
Where performance becomes excessively onerous due to fundamental change in circumstances, the disadvantaged party may request renegotiation. This is the UNIDROIT equivalent of force majeure for economic hardship — broader than Common Law frustration.

GROSS DISPARITY (Article 3.2.7):
A party may avoid a contract or individual term if it gave the other party an excessive advantage — taking into account the fact that the other party unfairly exploited its dependence, economic distress, or lack of experience.

INTERPRETATION (Articles 4.1-4.8):
Contracts are interpreted according to the common intention of the parties. Ambiguous terms are given their most reasonable meaning. Contradictions between standard terms and individually negotiated terms: the individually negotiated terms prevail.

INTEREST ON LATE PAYMENT (Article 7.4.9):
A party failing to pay money owes interest from the time payment was due at the average bank short-term lending rate for the currency of payment.""",
        },

        # ── EMPLOYMENT CONTRACT SPECIFICS ────────────────────────────────────
        {
            "clause_type": "general",
            "category":    "contract_type_guidance",
            "source":      "Employment Contract Standards — Common Law Jurisdictions",
            "risk_level":  "medium",
            "content": """EMPLOYMENT CONTRACTS — Key Analysis Points

Employment contracts require special analysis because employment law imposes mandatory minimum protections that cannot be contractually waived. Always analyse against applicable employment law, not just contract law principles.

MANDATORY MINIMUM PROVISIONS (Common Law Jurisdictions):
- Minimum wage requirements — no contract can pay below statutory minimum
- Working hours — maximum working week limitations apply
- Annual leave entitlement — minimum statutory leave must be provided
- Notice periods — minimum statutory notice requirements
- Discrimination protections — contractual terms cannot discriminate on protected grounds
- Health and safety obligations — employer cannot contract out of duty of care

KEY CLAUSES TO REVIEW:
1. Job title and duties: Vague or excessively broad "and other duties as required" language may be used to change the role significantly without agreement.
2. Remuneration: Is it clearly stated? Does it include or exclude benefits? What is the review mechanism?
3. Working hours: Are they reasonable? Is overtime paid or included?
4. Probationary period: Notice during probation is typically shorter — check duration and terms.
5. Post-termination restrictions: Non-compete, non-solicit, and garden leave provisions require careful analysis — they must be reasonable in scope, duration, and geography to be enforceable.
6. Intellectual property: All IP created during employment is typically owned by the employer — check scope and whether it extends to personal time.
7. Restrictive covenants: Post-employment restrictions are enforceable only if they protect a legitimate business interest and go no further than reasonably necessary.

RED FLAGS IN EMPLOYMENT CONTRACTS:
- Overly broad non-compete (more than 12 months, no geographic limit) — likely unenforceable but creates uncertainty.
- All IP assignment including work done in own time on own equipment — may be unenforceable depending on jurisdiction.
- Unilateral variation clause — employer cannot generally change fundamental terms without agreement.
- Payment in lieu of notice (PILON) clauses — check whether PILON is calculated on base salary only or full remuneration package.""",
        },

        # ── NDA SPECIFICS ────────────────────────────────────────────────────
        {
            "clause_type": "confidentiality",
            "category":    "contract_type_guidance",
            "source":      "NDA Analysis Standards",
            "risk_level":  "medium",
            "content": """NON-DISCLOSURE AGREEMENTS (NDAs) — Key Analysis Points

NDAs (also called Confidentiality Agreements) are standalone agreements whose primary purpose is to protect confidential information shared between parties. They are commonly used before business discussions, employment negotiations, M&A due diligence, and technology licensing.

MUTUAL vs ONE-WAY NDAs:
- Mutual NDA: Both parties share and receive confidential information — both are bound. Appropriate for exploratory business discussions.
- One-way NDA: Only one party discloses — only the recipient is bound. Appropriate where only one party has information to protect (e.g., a supplier briefing a potential customer).

KEY PROVISIONS:
1. Definition of Confidential Information: Must be precisely defined. Too narrow and important information is not protected. Too broad and it becomes unworkable.
2. Purpose limitation: Confidential information should only be usable for the specific permitted purpose — not for other commercial activities.
3. Standard of care: Recipient must protect information with "at least the same degree of care as it uses to protect its own confidential information, but no less than reasonable care."
4. Permitted disclosures: To employees and advisers on need-to-know basis — these should be bound by equivalent obligations.
5. Return/destruction: On request or termination, all confidential information and copies should be returned or destroyed.
6. Injunctive relief: Because damages may be inadequate for breach, NDAs typically provide that injunctive relief is an appropriate remedy.

RED FLAGS IN NDAs:
- No standard exclusions (public domain, prior knowledge, independent development, legal requirement to disclose).
- Perpetual duration for non-trade-secret information — 3-5 years is standard.
- No purpose limitation — information could be used for any purpose by the recipient.
- No permitted disclosure to advisers — impractical in real-world use.
- Automatic conversion of verbal disclosures into confidential information without a written summary confirmation within X days — creates uncertainty.""",
        },

        # ── RISK ASSESSMENT FRAMEWORK ────────────────────────────────────────
        {
            "clause_type": "general",
            "category":    "evaluation_framework",
            "source":      "LexAI Risk Assessment Methodology",
            "risk_level":  "low",
            "content": """LEXAI RISK ASSESSMENT FRAMEWORK

RISK LEVELS:
- CRITICAL: Clause creates unlimited exposure, waives fundamental rights, or is likely unenforceable in a way that removes intended protection. Immediate legal advice required.
- HIGH: Clause significantly favours the other party, creates substantial financial exposure, or removes important protections. Negotiation strongly recommended.
- MEDIUM: Clause is commercially unusual or contains provisions that could be problematic in certain circumstances. Review and consider negotiation.
- LOW: Clause is standard and generally balanced. Minor improvements may be possible but are not urgent.

ASSESSMENT CRITERIA:
1. Clause clarity: Is the obligation clearly defined? Ambiguous obligations favour the drafter (contra proferentem).
2. Commercial balance: Does the clause create obligations significantly more onerous for one party?
3. Enforceability: Is the clause likely to be enforceable in the governing jurisdiction?
4. Risk allocation: Does the clause allocate risk appropriately relative to each party's ability to manage it?
5. Missing protections: Are standard protective provisions absent?
6. Unusual provisions: Does the clause deviate significantly from market standard?

OVERALL CONTRACT RISK:
- CRITICAL: Multiple critical clauses or a pattern of one-sided provisions throughout.
- HIGH: One or more high-risk clauses with significant financial or operational impact.
- MEDIUM: Mixed provisions — some balanced, some concerning.
- LOW: Generally well-balanced agreement with standard provisions.

IMPORTANT DISCLAIMER:
This analysis is grounded in Common Law contract principles and internationally recognised standards. It is an AI-assisted analysis for informational purposes and does not constitute legal advice. For contractual matters with significant financial or operational consequences, engage a qualified legal practitioner in the relevant jurisdiction.""",
        },
    ]