# -*- coding: utf-8 -*-
"""
Production-ready SciTrans-LLMs GUI.
Clean, professional, fully functional.
"""

import gradio as gr
import os
from pathlib import Path
from scitran.extraction.pdf_parser import PDFParser
from scitran.core.pipeline import PipelineConfig, TranslationPipeline
from scitran.rendering.pdf_renderer import PDFRenderer
import tempfile


class ProductionGUI:
    """Production-ready translation GUI."""
    
    def __init__(self):
        self.config_file = Path.home() / ".scitrans" / "config.json"
        self.config_file.parent.mkdir(exist_ok=True)
        
        # Load config
        import json
        self.config = {}
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    self.config = json.load(f)
            except:
                pass
    
    def save_config(self):
        """Save configuration."""
        import json
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def translate_document(self, pdf_file, source_lang, target_lang, backend, 
                          enable_masking, enable_reranking):
        """Translate a PDF document."""
        if pdf_file is None:
            return "❌ Please upload a PDF file", None, ""
        
        try:
            # Get PDF path
            if hasattr(pdf_file, 'name'):
                input_path = Path(pdf_file.name)
            else:
                input_path = Path(pdf_file)
            
            status_msg = "🔄 Starting translation...\n"
            
            # Parse PDF
            status_msg += "📄 Parsing PDF...\n"
            parser = PDFParser()
            document = parser.parse(str(input_path), max_pages=None)
            total_blocks = sum(len(seg.blocks) for seg in document.segments)
            status_msg += f"✓ Parsed {total_blocks} blocks\n\n"
            
            # Configure pipeline
            status_msg += "⚙️ Configuring pipeline...\n"
            config = PipelineConfig(
                source_lang=source_lang,
                target_lang=target_lang,
                backend=backend,
                num_candidates=3 if enable_reranking else 1,
                enable_masking=enable_masking,
                enable_reranking=enable_reranking,
                quality_threshold=0.5
            )
            pipeline = TranslationPipeline(config)
            status_msg += f"✓ Using {backend} backend\n\n"
            
            # Translate
            status_msg += f"🔄 Translating {total_blocks} blocks...\n"
            result = pipeline.translate_document(document)
            status_msg += f"✓ Translated: {result.blocks_translated}\n"
            status_msg += f"✓ Failed: {result.blocks_failed}\n\n"
            
            # Render output
            status_msg += "📝 Rendering PDF...\n"
            output_path = input_path.parent / f"{input_path.stem}_translated.pdf"
            renderer = PDFRenderer()
            renderer.render_pdf(result.document, str(output_path))
            status_msg += f"✓ Saved to {output_path.name}\n\n"
            
            # Success message
            success_msg = "✅ Translation completed successfully!\n\n"
            success_msg += f"📊 Statistics:\n"
            success_msg += f"  • Backend: {backend}\n"
            success_msg += f"  • Blocks: {result.blocks_translated}/{total_blocks}\n"
            success_msg += f"  • Duration: {result.duration:.1f}s\n"
            if result.bleu_score:
                success_msg += f"  • BLEU: {result.bleu_score:.2f}\n"
            
            return success_msg, str(output_path), status_msg + success_msg
            
        except Exception as e:
            error_msg = f"❌ Translation failed\n\n"
            error_msg += f"Error: {str(e)}\n"
            import traceback
            detailed = traceback.format_exc()
            return error_msg, None, detailed
    
    def preview_pdf_page(self, pdf_file, page_num):
        """Preview a PDF page."""
        if pdf_file is None or page_num is None:
            return None
        
        try:
            import fitz
            from PIL import Image
            import io
            
            # Get PDF path
            if hasattr(pdf_file, 'name'):
                pdf_path = pdf_file.name
            else:
                pdf_path = pdf_file
            
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None
            
            page = doc[int(page_num) - 1]
            
            # Render to fit 700px width
            rect = page.rect
            zoom = 700 / rect.width
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            img_data = pix.tobytes("png")
            doc.close()
            
            img = Image.open(io.BytesIO(img_data))
            return img
            
        except Exception as e:
            print(f"Preview error: {e}")
            return None
    
    def test_backend(self, backend):
        """Test a backend."""
        try:
            config = PipelineConfig(
                source_lang='en',
                target_lang='fr',
                backend=backend,
                num_candidates=1,
                enable_masking=False,
                enable_reranking=False
            )
            pipeline = TranslationPipeline(config)
            
            # Test simple translation
            from scitran.core.models import Block, BlockType
            from datetime import datetime
            
            test_block = Block(
                block_id="test_0",
                source_text="Machine learning is a subset of artificial intelligence.",
                block_type=BlockType.PARAGRAPH
            )
            
            # Create minimal document for translation
            from scitran.core.models import Document, Segment
            segment = Segment(
                segment_id="seg_0",
                segment_type="body",
                blocks=[test_block]
            )
            doc = Document(
                document_id="test_doc",
                segments=[segment],
                source_path="test"
            )
            
            result = pipeline.translate_document(doc)
            
            if result.success and test_block.translated_text:
                output = f"✅ {backend.upper()} Backend Working!\n\n"
                output += f"Test translation:\n"
                output += f"EN: {test_block.source_text}\n"
                output += f"FR: {test_block.translated_text}\n"
                return output
            else:
                return f"⚠️ {backend.upper()} returned no translation"
                
        except Exception as e:
            return f"❌ {backend.upper()} Error:\n{str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        
        # Modern minimal theme
        theme = gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
            font=["Inter", "sans-serif"]
        )
        
        with gr.Blocks(
            theme=theme,
            title="SciTrans-LLMs",
            css="""
            .main-container {max-width: 1400px; margin: 0 auto;}
            .compact-row {gap: 0.5rem !important;}
            .status-box {font-family: monospace; font-size: 0.9rem;}
            .preview-container {border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem;}
            """
        ) as demo:
            
            gr.Markdown(
                """
                # 🌐 SciTrans-LLMs
                **Professional Scientific Document Translation**
                
                Translate academic PDFs with LaTeX preservation, quality reranking, and layout preservation.
                """
            )
            
            with gr.Tabs():
                # TAB 1: TRANSLATION
                with gr.Tab("📄 Translation"):
                    with gr.Row():
                        # LEFT: Configuration (55%)
                        with gr.Column(scale=55):
                            gr.Markdown("### ⚙️ Configuration")
                            
                            pdf_upload = gr.File(
                                label="📎 Upload PDF",
                                file_types=[".pdf"],
                                type="file"
                            )
                            
                            with gr.Row():
                                source_lang = gr.Dropdown(
                                    choices=["en", "fr", "de", "es", "it", "pt", "zh", "ja", "ko"],
                                    value="en",
                                    label="Source Language",
                                    scale=1
                                )
                                target_lang = gr.Dropdown(
                                    choices=["en", "fr", "de", "es", "it", "pt", "zh", "ja", "ko"],
                                    value="fr",
                                    label="Target Language",
                                    scale=1
                                )
                            
                            backend = gr.Dropdown(
                                choices=["cascade", "free", "ollama", "deepseek", "openai", "anthropic"],
                                value="cascade",
                                label="Translation Backend"
                            )
                            
                            gr.Markdown("**Quick Options**")
                            with gr.Row():
                                enable_masking = gr.Checkbox(
                                    value=True,
                                    label="🎭 LaTeX Masking",
                                    scale=1
                                )
                                enable_reranking = gr.Checkbox(
                                    value=False,
                                    label="🏆 Quality Reranking",
                                    scale=1
                                )
                            
                            translate_btn = gr.Button(
                                "🚀 Translate Document",
                                variant="primary",
                                size="lg"
                            )
                            
                            status_box = gr.Textbox(
                                label="Status",
                                lines=6,
                                interactive=False,
                                elem_classes=["status-box"]
                            )
                            
                            output_file = gr.File(
                                label="📥 Download Translated PDF"
                            )
                        
                        # RIGHT: Preview (45%)
                        with gr.Column(scale=45):
                            gr.Markdown("### 👁️ PDF Preview")
                            
                            with gr.Group(elem_classes=["preview-container"]):
                                pdf_preview_image = gr.Image(
                                    label="",
                                    show_label=False,
                                    height=600
                                )
                                
                                # Compact pagination
                                with gr.Row(elem_classes=["compact-row"]):
                                    prev_btn = gr.Button("◀", size="sm", scale=1)
                                    page_num = gr.Number(
                                        value=1,
                                        minimum=1,
                                        label="Page",
                                        scale=2
                                    )
                                    page_total = gr.Textbox(
                                        value="/ 1",
                                        show_label=False,
                                        interactive=False,
                                        scale=1
                                    )
                                    next_btn = gr.Button("▶", size="sm", scale=1)
                    
                    # Log accordion
                    with gr.Accordion("📋 Detailed Log", open=False):
                        detailed_log = gr.Textbox(
                            lines=10,
                            interactive=False,
                            elem_classes=["status-box"]
                        )
                
                # TAB 2: TESTING
                with gr.Tab("🧪 Testing"):
                    gr.Markdown("### Test Translation Backends")
                    
                    with gr.Row():
                        with gr.Column():
                            test_backend_select = gr.Dropdown(
                                choices=["cascade", "free", "ollama", "deepseek", "openai", "anthropic"],
                                value="cascade",
                                label="Select Backend"
                            )
                            test_btn = gr.Button("🧪 Test Backend", variant="primary")
                        
                        with gr.Column():
                            test_output = gr.Textbox(
                                label="Test Results",
                                lines=8,
                                interactive=False
                            )
                
                # TAB 3: ABOUT
                with gr.Tab("ℹ️ About"):
                    gr.Markdown("""
                    ## SciTrans-LLMs
                    
                    **Professional scientific document translation with:**
                    
                    - 🎭 **LaTeX Masking** - Preserves equations and formulas
                    - 🏆 **Quality Reranking** - Multi-candidate selection
                    - 📐 **Layout Preservation** - Maintains document structure
                    
                    ### Available Backends
                    
                    | Backend | Cost | Quality | Speed |
                    |---------|------|---------|-------|
                    | CASCADE | FREE | Good | Fast |
                    | FREE | FREE | Good | Fast |
                    | Ollama | FREE | Good | Medium |
                    | DeepSeek | $ | Excellent | Fast |
                    | OpenAI | $$$ | Excellent | Fast |
                    | Anthropic | $$$ | Excellent | Medium |
                    
                    ### Usage
                    
                    1. Upload PDF
                    2. Select languages
                    3. Choose backend
                    4. Click translate
                    5. Download result
                    
                    **CLI Alternative:**
                    ```bash
                    scitrans translate paper.pdf --backend cascade
                    ```
                    """)
            
            # Event handlers
            def handle_upload(pdf):
                if pdf is None:
                    return None, 1, "/ 1"
                try:
                    import fitz
                    doc = fitz.open(pdf.name)
                    num_pages = len(doc)
                    doc.close()
                    preview = self.preview_pdf_page(pdf, 1)
                    return preview, 1, f"/ {num_pages}"
                except:
                    return None, 1, "/ 1"
            
            def next_page(pdf, current_page, total_text):
                if pdf is None:
                    return None, current_page
                try:
                    max_page = int(total_text.split()[-1])
                    new_page = min(max_page, int(current_page) + 1)
                    preview = self.preview_pdf_page(pdf, new_page)
                    return preview, new_page
                except:
                    return None, current_page
            
            def prev_page(pdf, current_page):
                if pdf is None:
                    return None, current_page
                new_page = max(1, int(current_page) - 1)
                preview = self.preview_pdf_page(pdf, new_page)
                return preview, new_page
            
            # Connect events
            pdf_upload.change(
                fn=handle_upload,
                inputs=[pdf_upload],
                outputs=[pdf_preview_image, page_num, page_total]
            )
            
            page_num.change(
                fn=self.preview_pdf_page,
                inputs=[pdf_upload, page_num],
                outputs=[pdf_preview_image]
            )
            
            next_btn.click(
                fn=next_page,
                inputs=[pdf_upload, page_num, page_total],
                outputs=[pdf_preview_image, page_num]
            )
            
            prev_btn.click(
                fn=prev_page,
                inputs=[pdf_upload, page_num],
                outputs=[pdf_preview_image, page_num]
            )
            
            translate_btn.click(
                fn=self.translate_document,
                inputs=[pdf_upload, source_lang, target_lang, backend, enable_masking, enable_reranking],
                outputs=[status_box, output_file, detailed_log]
            )
            
            test_btn.click(
                fn=self.test_backend,
                inputs=[test_backend_select],
                outputs=[test_output]
            )
        
        return demo


def launch_gui():
    """Launch the production GUI."""
    import os
    import signal
    import subprocess
    import warnings
    
    # Auto-kill port 7860
    try:
        result = subprocess.run(
            ["lsof", "-ti", ":7860"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except:
                    pass
            import time
            time.sleep(1)
    except:
        pass
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    app = ProductionGUI()
    interface = app.create_interface()
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    launch_gui()
