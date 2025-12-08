"""Simplified GUI for SciTrans-LLMs with minimal dependencies."""

import gradio as gr
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scitran.core.pipeline import TranslationPipeline, PipelineConfig


def translate_document(
    pdf_file,
    backend,
    source_lang,
    target_lang,
    enable_masking,
    enable_reranking
):
    """Simple translation function."""
    if pdf_file is None:
        return "❌ Please upload a PDF file", None
    
    try:
        # Configure pipeline
        config = PipelineConfig(
            source_lang=source_lang,
            target_lang=target_lang,
            backend=backend,
            enable_masking=enable_masking,
            enable_reranking=enable_reranking
        )
        
        # Create pipeline
        pipeline = TranslationPipeline(config)
        
        # Get output path
        input_path = Path(pdf_file.name)
        output_path = input_path.parent / f"{input_path.stem}_translated.pdf"
        
        # Translate
        status_msg = f"🔄 Translating with {backend} backend..."
        result = pipeline.translate_pdf(str(input_path), str(output_path))
        
        if result:
            success_msg = f"✅ Translation completed!\n\nBackend: {backend}\nOutput: {output_path.name}"
            return success_msg, str(output_path)
        else:
            return "❌ Translation failed", None
            
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def create_gui():
    """Create simplified Gradio interface."""
    
    with gr.Blocks(title="SciTrans-LLMs", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 SciTrans-LLMs: Scientific Document Translation
        
        Translate scientific PDFs preserving LaTeX formulas and formatting.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Configuration")
                
                # Backend selection
                backend = gr.Dropdown(
                    choices=["cascade", "free", "ollama", "deepseek", "openai", "anthropic"],
                    value="cascade",
                    label="Translation Backend",
                    info="💡 cascade and free are FREE! No API key needed."
                )
                
                # Language settings
                with gr.Row():
                    source_lang = gr.Textbox(
                        value="en",
                        label="Source Language",
                        placeholder="en"
                    )
                    target_lang = gr.Textbox(
                        value="fr",
                        label="Target Language",
                        placeholder="fr"
                    )
                
                # Options
                enable_masking = gr.Checkbox(
                    value=True,
                    label="Enable LaTeX Masking",
                    info="Protects formulas and code blocks"
                )
                
                enable_reranking = gr.Checkbox(
                    value=False,
                    label="Enable Quality Reranking",
                    info="Generates multiple candidates and selects best"
                )
                
            with gr.Column(scale=3):
                gr.Markdown("### 📄 Document")
                
                # File upload
                pdf_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"]
                )
                
                # Translate button
                translate_btn = gr.Button(
                    "🔄 Translate Document",
                    variant="primary",
                    size="lg"
                )
                
                # Status output
                status_output = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False
                )
                
                # Download output
                file_output = gr.File(
                    label="Download Translated PDF",
                    interactive=False
                )
        
        # Backend info
        with gr.Accordion("ℹ️ Backend Information", open=False):
            gr.Markdown("""
            ### Available Backends
            
            | Backend | Cost | API Key | Quality | Speed | Best For |
            |---------|------|---------|---------|-------|----------|
            | **cascade** | FREE | ❌ No | ⭐⭐⭐ | Fast | Testing, Learning |
            | **free** | FREE | ❌ No | ⭐⭐⭐ | Fast | Quick translations |
            | **ollama** | FREE | ❌ No | ⭐⭐⭐⭐ | Medium | Offline, Privacy |
            | **deepseek** | $ | ✅ Yes | ⭐⭐⭐⭐ | Fast | Cost-effective |
            | **openai** | $$$ | ✅ Yes | ⭐⭐⭐⭐⭐ | Fast | Best quality |
            | **anthropic** | $$$ | ✅ Yes | ⭐⭐⭐⭐⭐ | Medium | Long documents |
            
            ### API Keys Setup
            
            For paid backends, set environment variables:
            ```bash
            export OPENAI_API_KEY="sk-your-key"
            export ANTHROPIC_API_KEY="sk-ant-your-key"
            export DEEPSEEK_API_KEY="your-key"
            ```
            
            See `API_KEYS_SETUP.md` for detailed instructions.
            """)
        
        # Connect button to function
        translate_btn.click(
            fn=translate_document,
            inputs=[
                pdf_input,
                backend,
                source_lang,
                target_lang,
                enable_masking,
                enable_reranking
            ],
            outputs=[status_output, file_output]
        )
        
        gr.Markdown("""
        ---
        
        ### 💡 Quick Tips
        
        - **cascade** backend is recommended for FREE, reliable translations
        - Enable **LaTeX Masking** for scientific documents
        - Use **Reranking** for best quality (slower but better)
        - Check `API_KEYS_SETUP.md` for paid backend setup
        
        ### 📚 Documentation
        
        - `README.md` - Project overview
        - `API_KEYS_SETUP.md` - API key setup guide
        - `FINAL_STATUS.md` - Current system status
        
        **Version 2.0.0** | Built with ❤️ for scientific research
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_gui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
