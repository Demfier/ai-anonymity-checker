# Anonymization Checker

Check academic papers for double-blind anonymity policy violations. Uses comprehensive LLM analysis to detect identifying information while minimizing false positives.

## Features

- **12 anonymization checks** covering author names, affiliations, self-citations, acknowledgments, code repos, contact info, PDF metadata, LaTeX artifacts, de-anonymization leftovers, funding, arXiv self-references, and watermarks/headers
- **Multiple LLM backends**: OpenAI, OpenRouter, Ollama, vLLM, or any OpenAI-compatible API
- **Native multimodal capabilities**: Pushes PDFs natively to OpenRouter API to skip third-party parsing limits, seamlessly catching metadata and hidden text
- **Both PDF and LaTeX** input formats
- **CLI and Web UI** (Gradio)
- **JSON and Markdown** reports with evidence, confidence scores, and remediation suggestions

## Installation

```bash
# Core
uv sync

# With LLM support / formatting / Web UI
uv sync --all-extras
```

## Quick Start

### Running Checks

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
uv run anonymization-checker check paper.pdf -p openai -m gpt-4o

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...
uv run anonymization-checker check paper.pdf -p openrouter -m anthropic/claude-sonnet-4

# Ollama (local)
uv run anonymization-checker check paper.pdf -p ollama -m llama3.1

# vLLM (local)
uv run anonymization-checker check paper.pdf -p vllm -m default --base-url http://localhost:8000/v1

# Any OpenAI-compatible API
uv run anonymization-checker check paper.pdf -p custom --base-url https://my-api.com/v1 -k my-key -m my-model
```

### Web UI

```bash
uv run python -m anonymization_checker.web
# Opens at http://localhost:7860
```

### Python API

```python
from anonymization_checker.checker import AnonymizationChecker
from anonymization_checker.config import AppConfig, LLMConfig, LLMProvider

config = AppConfig(
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
        api_key="sk-...",
    )
)

checker = AnonymizationChecker(config)

# JSON report
report = checker.check_file("paper.pdf")
print(report["summary"]["recommendation"])  # PASS, MINOR_ISSUES, REVIEW_REQUIRED, or FAIL

# Markdown report
md = checker.check_file_markdown("paper.pdf")
print(md)
```

## CLI Options

```
uv run anonymization-checker check <file> [options]

Arguments:
  file                  Path to PDF or LaTeX file

Options:
  -p, --provider        LLM provider: openai, openrouter, ollama, vllm, custom
  -m, --model           Model name (defaults to provider's default)
  -k, --api-key         API key (or use env vars)
  --base-url            Custom API base URL
  -f, --format          Output: json, markdown, both (default: markdown)
  -o, --output          Save report to file
  -v, --verbose         Verbose logging
```

## Checks

| Check | Severity | What it detects |
|-------|----------|-----------------|
| `author_names` | Critical | Author names in body text (not references) |
| `self_citations` | Critical | "Our previous work [X]" self-citation patterns |
| `affiliations` | High | "Our lab at MIT" institutional mentions |
| `acknowledgments` | High | Named people, specific grants in acknowledgments |
| `code_repos` | High | GitHub URLs with identifying usernames |
| `arxiv_self_refs` | High | ArXiv preprint self-references |
| `contact_info` | High | Emails and phone numbers |
| `pdf_metadata` | High | Author/creator in PDF metadata fields |
| `latex_artifacts` | High | LaTeX comments, \author{}, file paths |
| `funding` | Medium | Traceable grant numbers (NSF, NIH, etc.) |
| `deanon_leftovers` | Medium | "Camera-ready", "author version" text |
| `watermarks_headers` | Medium | Institution names in headers/footers |

## Exit Codes

- `0` — PASS or MINOR_ISSUES
- `1` — REVIEW_REQUIRED (high-severity violations)
- `2` — FAIL (critical violations)

## Architecture

All LLM providers use the same OpenAI-compatible chat completions API. The `LLMClient` class accepts a `base_url` parameter — switching providers just means changing the URL:

- OpenAI: `https://api.openai.com/v1`
- OpenRouter: `https://openrouter.ai/api/v1`
- Ollama: `http://localhost:11434/v1`
- vLLM: `http://localhost:8000/v1`

If the `openai` Python package is installed, it's used for HTTP requests. Otherwise, the client falls back to stdlib `urllib` — no external dependencies required for the core functionality.

## License

MIT
