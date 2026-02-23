"""Gradio web interface for the anonymization checker."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .checker import AnonymizationChecker
from .config import AppConfig, LLMConfig, LLMProvider


def create_app():
    import gradio as gr

    def run_check(file, provider: str, model: str, api_key: str, base_url: str):
        if file is None:
            return "Please upload a PDF or LaTeX file.", "{}"

        file_path = Path(file.name if hasattr(file, "name") else file)
        if file_path.suffix.lower() not in (".pdf", ".tex", ".latex"):
            return "Unsupported file type. Upload .pdf, .tex, or .latex.", "{}"

        try:
            llm_provider = LLMProvider(provider.lower())
        except ValueError:
            llm_provider = LLMProvider.CUSTOM

        config = AppConfig(
            llm=LLMConfig(
                provider=llm_provider,
                model=model or "",
                api_key=api_key or "",
                base_url=base_url or "",
            ),
        )

        checker = AnonymizationChecker(config)
        try:
            report = checker.check_file(file_path)
            md = checker._report_to_markdown(report)
            return md, json.dumps(report, indent=2)
        except Exception as e:
            logging.error(f"Check failed: {e}", exc_info=True)
            return f"Error: {e}", "{}"

    with gr.Blocks(title="Anonymization Checker", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🔍 Anonymization Checker\n"
            "Upload a paper to check for double-blind anonymity violations."
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload Paper",
                    file_types=[".pdf", ".tex", ".latex"],
                )
                provider = gr.Dropdown(
                    choices=["openai", "openrouter", "ollama", "vllm", "custom"],
                    value="openai", label="Provider",
                )
                model = gr.Textbox(label="Model", placeholder="Leave empty for default")
                api_key = gr.Textbox(label="API Key", type="password",
                                     placeholder="Or set env var")
                base_url = gr.Textbox(label="Base URL (optional)")
                check_btn = gr.Button("🔍 Check Paper", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Report"):
                        md_out = gr.Markdown(value="Upload a paper to start.")
                    with gr.Tab("JSON"):
                        json_out = gr.Code(language="json")

        check_btn.click(
            fn=run_check,
            inputs=[file_input, provider, model, api_key, base_url],
            outputs=[md_out, json_out],
        )

    return demo


def main():
    demo = create_app()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
