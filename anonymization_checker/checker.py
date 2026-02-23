"""Anonymization checker — single LLM prompt approach.

Extracts text from PDF/LaTeX, sends it to an LLM with a comprehensive prompt
that covers all 12 anonymization violation types, and returns structured results.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, LLMConfig
from .extractors.latex_extractor import LaTeXExtractor
from .extractors.pdf_extractor import PDFExtractor
from .llm.base import LLMClient
from .llm.factory import create_llm_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    """Load .env file into os.environ. Searches cwd and parent dirs."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
        return
    except ImportError:
        pass

    search = Path.cwd()
    for _ in range(4):
        env_file = search / ".env"
        if env_file.is_file():
            try:
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            except Exception:
                pass
            return
        if search.parent == search:
            break
        search = search.parent


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert academic paper anonymity reviewer. Your job is to check whether
a paper submitted for double-blind peer review contains ANY information that could
reveal the authors' identity. Be thorough but precise — only flag genuine violations,
not false positives."""

ANALYSIS_PROMPT = """\
Analyze the following paper text for double-blind anonymity violations.

Check for ALL of the following violation types:

1. **Author names in body text** — Author names appearing outside the references
   section (e.g., in headers, footers, the body, or as "Author: X").
   DO NOT FLAG: names that appear only inside citations/references like
   "[Smith et al., 2020]" or in the bibliography — those are fine.

2. **Institutional affiliations** — Mentions that tie the work to a specific
   institution with first-person language (e.g., "our lab at MIT",
   "we at Stanford").
   DO NOT FLAG: generic discussion of institutions in related work, or
   phrases like "a top-tier university" without naming it.

3. **Self-citations revealing authorship** — Citations combined with first-person
   language that claim authorship of the cited work (e.g., "In our previous
   work [5]", "Building on our earlier framework [Smith, 2022]",
   "We showed in [3] that...").
   DO NOT FLAG: passive citations ("It was shown in [5]"), third-person
   citations ("Smith et al. [5] showed"), or citing well-known work.

4. **Acknowledgments with identifying info** — Named individuals (advisors,
   collaborators), specific traceable grant numbers (e.g., "NSF CAREER
   #1234567", "NIH R01-GM123456"), or named labs/groups in an acknowledgments
   section.
   DO NOT FLAG: "We thank the anonymous reviewers" or generic funder names
   without specific award numbers.

5. **Code repository URLs** — GitHub/GitLab/Bitbucket URLs with usernames or
   organization names that identify the authors (e.g.,
   "github.com/jsmith/myproject").
   DO NOT FLAG: well-known public repos (pytorch, tensorflow, huggingface).

6. **Contact information** — Email addresses (user@domain.com) or phone numbers
   in the body text.
   DO NOT FLAG: metric@k notation (NDCG@8, Recall@50), example emails
   (test@example.com), or emails only in the references.

7. **PDF/document metadata** — Author names or personal identifiers in document
   metadata fields (Author, Creator, etc. if visible in the text).

8. **LaTeX artifacts** — \\author{{}}, \\affiliation{{}}, \\email{{}} commands with
   real content; comments containing author names or TODOs with names;
   file paths with usernames (/home/alice/..., C:\\Users\\Bob\\...).

9. **De-anonymization leftovers** — Phrases like "camera-ready version",
   "de-anonymized", "author version", "non-anonymous version", or
   "Author(s):" labels.

10. **Traceable funding** — Specific grant or contract numbers that can be looked
    up to identify the PI (e.g., "NSF #1234567", "ERC-2020-STG-123456").
    DO NOT FLAG: generic funding mentions ("supported by NSF") without numbers.

11. **ArXiv/preprint self-references** — ArXiv URLs referenced with first-person
    language claiming authorship ("our preprint [arxiv:2301.12345]").
    DO NOT FLAG: citing others' arXiv papers.

12. **Unique project/dataset/lab names** — Names so specific they identify the
    authors (e.g., "the SmithLab Benchmark", "our ACME framework").
    DO NOT FLAG: well-known public benchmarks (ImageNet, GLUE, etc.).

---

Respond with a JSON object in this exact format:

{{
  "violations": [
    {{
      "type": "<one of: author_names, affiliations, self_citations, acknowledgments, code_repos, contact_info, metadata, latex_artifacts, deanon_leftovers, funding, arxiv_self_refs, project_names>",
      "severity": "<critical | high | medium | low>",
      "text": "<the exact offending text from the paper>",
      "reason": "<one sentence explaining why this is a violation>",
      "remediation": "<one sentence on how to fix it>"
    }}
  ],
  "summary": "<1-2 sentence overall assessment>"
}}

If the paper has NO violations, return:
{{"violations": [], "summary": "No anonymity violations detected."}}

Severity guide:
- critical: directly reveals author identity (names, emails, self-citations)
- high: strongly suggests identity (affiliations, code repos, acknowledgments)
- medium: could potentially identify (funding numbers, arXiv self-refs, project names)
- low: minor risk (metadata, leftovers)

---

PAPER TEXT:

{paper_text}"""

PDF_ANALYSIS_PROMPT = """\
Analyze the attached paper PDF for double-blind anonymity violations.

Check for ALL of the following violation types:

1. **Author names in body text** — Author names appearing outside the references
   section (e.g., in headers, footers, the body, or as "Author: X").
   DO NOT FLAG: names that appear only inside citations/references like
   "[Smith et al., 2020]" or in the bibliography — those are fine.

2. **Institutional affiliations** — Mentions that tie the work to a specific
   institution with first-person language (e.g., "our lab at MIT",
   "we at Stanford").
   DO NOT FLAG: generic discussion of institutions in related work, or
   phrases like "a top-tier university" without naming it.

3. **Self-citations revealing authorship** — Citations combined with first-person
   language that claim authorship of the cited work (e.g., "In our previous
   work [5]", "Building on our earlier framework [Smith, 2022]",
   "We showed in [3] that...").
   DO NOT FLAG: passive citations ("It was shown in [5]"), third-person
   citations ("Smith et al. [5] showed"), or citing well-known work.

4. **Acknowledgments with identifying info** — Named individuals (advisors,
   collaborators), specific traceable grant numbers (e.g., "NSF CAREER
   #1234567", "NIH R01-GM123456"), or named labs/groups in an acknowledgments
   section.
   DO NOT FLAG: "We thank the anonymous reviewers" or generic funder names
   without specific award numbers.

5. **Code repository URLs** — GitHub/GitLab/Bitbucket URLs with usernames or
   organization names that identify the authors (e.g.,
   "github.com/jsmith/myproject").
   DO NOT FLAG: well-known public repos (pytorch, tensorflow, huggingface).

6. **Contact information** — Email addresses (user@domain.com) or phone numbers
   in the body text.
   DO NOT FLAG: metric@k notation (NDCG@8, Recall@50), example emails
   (test@example.com), or emails only in the references.

7. **PDF/document metadata** — Author names or personal identifiers in document
   metadata fields. OpenRouter's native file parsing captures PDF metadata, so please pay attention to any metadata fields returned.

8. **LaTeX artifacts** — \\author{{}}, \\affiliation{{}}, \\email{{}} commands with
   real content; comments containing author names or TODOs with names;
   file paths with usernames (/home/alice/..., C:\\Users\\Bob\\...).

9. **De-anonymization leftovers** — Phrases like "camera-ready version",
   "de-anonymized", "author version", "non-anonymous version", or
   "Author(s):" labels.

10. **Traceable funding** — Specific grant or contract numbers that can be looked
    up to identify the PI (e.g., "NSF #1234567", "ERC-2020-STG-123456").
    DO NOT FLAG: generic funding mentions ("supported by NSF") without numbers.

11. **ArXiv/preprint self-references** — ArXiv URLs referenced with first-person
    language claiming authorship ("our preprint [arxiv:2301.12345]").
    DO NOT FLAG: citing others' arXiv papers.

12. **Unique project/dataset/lab names** — Names so specific they identify the
    authors (e.g., "the SmithLab Benchmark", "our ACME framework").
    DO NOT FLAG: well-known public benchmarks (ImageNet, GLUE, etc.).

---

Respond with a JSON object in this exact format:

{{
  "violations": [
    {{
      "type": "<one of: author_names, affiliations, self_citations, acknowledgments, code_repos, contact_info, metadata, latex_artifacts, deanon_leftovers, funding, arxiv_self_refs, project_names>",
      "severity": "<critical | high | medium | low>",
      "text": "<the exact offending text from the paper>",
      "reason": "<one sentence explaining why this is a violation>",
      "remediation": "<one sentence on how to fix it>"
    }}
  ],
  "summary": "<1-2 sentence overall assessment>"
}}

If the paper has NO violations, return:
{{"violations": [], "summary": "No anonymity violations detected."}}

Severity guide:
- critical: directly reveals author identity (names, emails, self-citations)
- high: strongly suggests identity (affiliations, code repos, acknowledgments)
- medium: could potentially identify (funding numbers, arXiv self-refs, project names)
- low: minor risk (metadata, leftovers)

"""


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class AnonymizationChecker:
    """Check papers for anonymity violations using a single LLM call."""

    def __init__(self, config: AppConfig | None = None):
        _load_dotenv()
        self.config = config or AppConfig()
        self._llm: LLMClient | None = None
        self._pdf_extractor = PDFExtractor()
        self._latex_extractor = LaTeXExtractor()

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = create_llm_client(self.config.llm)
        return self._llm

    def check_file(self, file_path: str | Path) -> dict[str, Any]:
        """Run anonymization check and return the full report dict."""
        file_path = Path(file_path)
        start = time.time()

        content = self._prepare_content(file_path)
        llm_result = self._analyze(content)
        elapsed = time.time() - start

        return self._build_report(llm_result, str(file_path), elapsed)

    def check_file_markdown(self, file_path: str | Path) -> str:
        """Run anonymization check and return a Markdown report."""
        report = self.check_file(file_path)
        return self._report_to_markdown(report)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare_content(self, file_path: str | Path) -> str | list[dict[str, Any]]:
        """Prepare content from PDF or LaTeX for LLM submission."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            # For PDFs, use openrouter multimodal file input
            with open(file_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
            data_url = f"data:application/pdf;base64,{base64_pdf}"

            return [
                {"type": "text", "text": PDF_ANALYSIS_PROMPT},
                {
                    "type": "file",
                    "file": {"filename": file_path.name, "file_data": data_url},
                },
            ]
        elif suffix in (".tex", ".latex"):
            doc = self._latex_extractor.extract(file_path)
            # For LaTeX, send the raw source — the LLM can read LaTeX
            return doc.full_text
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. Use .pdf, .tex, or .latex"
            )

    def _analyze(self, content: str | list[dict[str, Any]]) -> dict[str, Any]:
        """Send paper text or file list to LLM and get violation analysis."""
        prompt = content
        if isinstance(content, str):
            # Truncate if too long (most LLMs have context limits)
            max_chars = 60_000
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[... truncated for length ...]"
            prompt = ANALYSIS_PROMPT.format(paper_text=content)

        result = self.llm.complete_json(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
        )

        if not isinstance(result, dict):
            result = {"violations": [], "summary": "Unexpected response format."}

        return result

    def _build_report(
        self, llm_result: dict[str, Any], file_path: str, elapsed: float
    ) -> dict[str, Any]:
        """Build the full structured report from LLM output."""
        violations = llm_result.get("violations", [])
        summary_text = llm_result.get("summary", "")
        error = llm_result.get("error", "")

        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for v in violations:
            sev = v.get("severity", "medium")
            if sev in severity_counts:
                severity_counts[sev] += 1

        total = len(violations)
        if total == 0:
            recommendation = "PASS"
        elif severity_counts["critical"] > 0:
            recommendation = "FAIL"
        elif severity_counts["high"] > 0:
            recommendation = "REVIEW_REQUIRED"
        else:
            recommendation = "MINOR_ISSUES"

        return {
            "metadata": {
                "file_path": file_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": self.config.llm.provider.value,
                "model": self.config.llm.get_model(),
                "execution_time_seconds": round(elapsed, 2),
            },
            "summary": {
                "total_violations": total,
                "severity_breakdown": severity_counts,
                "recommendation": recommendation,
                "assessment": summary_text,
            },
            "violations": violations,
            "error": error,
        }

    @staticmethod
    def _report_to_markdown(report: dict[str, Any]) -> str:
        """Convert a report dict to a Markdown string."""
        lines: list[str] = []
        meta = report["metadata"]
        summary = report["summary"]
        violations = report["violations"]

        lines.append("# Anonymization Check Report")
        lines.append("")
        lines.append(f"**File:** `{meta['file_path']}`")
        lines.append(f"**Model:** {meta['provider']} / {meta['model']}")
        lines.append(f"**Time:** {meta['execution_time_seconds']}s")
        lines.append("")

        rec = summary["recommendation"]
        badge = {
            "PASS": "✅ PASS",
            "MINOR_ISSUES": "💡 MINOR ISSUES",
            "REVIEW_REQUIRED": "⚠️ REVIEW REQUIRED",
            "FAIL": "❌ FAIL",
        }
        lines.append(f"## {badge.get(rec, rec)}")
        lines.append("")
        if summary["assessment"]:
            lines.append(summary["assessment"])
            lines.append("")

        sc = summary["severity_breakdown"]
        lines.append(f"| Severity | Count |")
        lines.append(f"|----------|-------|")
        lines.append(f"| 🔴 Critical | {sc['critical']} |")
        lines.append(f"| 🟠 High | {sc['high']} |")
        lines.append(f"| 🟡 Medium | {sc['medium']} |")
        lines.append(f"| 🔵 Low | {sc['low']} |")
        lines.append("")

        if report.get("error"):
            lines.append(f"**Error:** {report['error']}")
            lines.append("")

        if not violations:
            lines.append("No violations found.")
            return "\n".join(lines)

        lines.append("---")
        lines.append("")

        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        violations_sorted = sorted(
            violations, key=lambda v: sev_order.get(v.get("severity", "medium"), 9)
        )

        sev_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}

        for i, v in enumerate(violations_sorted, 1):
            sev = v.get("severity", "medium")
            emoji = sev_emoji.get(sev, "⚪")
            vtype = v.get("type", "unknown")
            lines.append(f"### {emoji} {sev.upper()} — {vtype}")
            lines.append("")
            lines.append(f"> {v.get('text', '')}")
            lines.append("")
            lines.append(f"**Why:** {v.get('reason', '')}")
            lines.append("")
            lines.append(f"**Fix:** {v.get('remediation', '')}")
            lines.append("")

        return "\n".join(lines)
