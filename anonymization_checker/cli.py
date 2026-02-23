"""CLI interface for the anonymization checker."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .checker import AnonymizationChecker
from .config import AppConfig, LLMConfig, LLMProvider


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="anonymization-checker",
        description="Check academic papers for double-blind anonymity violations.",
    )
    parser.add_argument("file", help="Path to PDF or LaTeX file")
    parser.add_argument("-p", "--provider", default="openai",
                        help="LLM provider: openai, openrouter, ollama, vllm, custom")
    parser.add_argument("-m", "--model", default="",
                        help="Model name (defaults per provider)")
    parser.add_argument("-k", "--api-key", default="",
                        help="API key (or set OPENAI_API_KEY / OPENROUTER_API_KEY)")
    parser.add_argument("--base-url", default="",
                        help="Custom API base URL")
    parser.add_argument("-f", "--format", default="markdown",
                        choices=["json", "markdown"],
                        help="Output format (default: markdown)")
    parser.add_argument("-o", "--output", default="",
                        help="Save report to file (omit for stdout)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    path = Path(args.file)
    if not path.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        llm_provider = LLMProvider(args.provider)
    except ValueError:
        llm_provider = LLMProvider.CUSTOM

    config = AppConfig(
        llm=LLMConfig(
            provider=llm_provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        ),
    )

    checker = AnonymizationChecker(config)

    print("Analyzing paper...", file=sys.stderr)
    report = checker.check_file(path)

    if args.format == "json":
        output = json.dumps(report, indent=2)
    else:
        output = checker._report_to_markdown(report)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Report saved to: {args.output}", file=sys.stderr)
    else:
        print(output)

    rec = report["summary"]["recommendation"]
    if rec == "FAIL":
        sys.exit(2)
    elif rec == "REVIEW_REQUIRED":
        sys.exit(1)


if __name__ == "__main__":
    main()
