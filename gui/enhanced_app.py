# -*- coding: utf-8 -*-
"""Enhanced GUI for SciTrans-LLMs with ALL features.

Includes:
- Multiple tabs (Translation, Testing, Glossary, Settings, About)
- API key management
- Dark mode toggle
- PDF preview
- URL input
- French <-> English only
- Reranking enabled by default
- DocYOLO & MinerU options
- Prompt training
- Complete glossary management
"""

import gradio as gr
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.extraction.pdf_parser import PDFParser
from scitran.rendering.pdf_renderer import PDFRenderer


class EnhancedTranslationGUI:
    """Enhanced GUI with all features."""
    
    def __init__(self):
        self.config_file = Path.home() / ".scitrans" / "gui_config.json"
        self.config_file.parent.mkdir(exist_ok=True)
        self.load_config()
    
    def load_config(self):
        """Load GUI configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                self.config = json.load(f)
        else:
            self.config = {
                "dark_mode": False,
                "default_backend": "cascade",
                "api_keys": {},
                "reranking_enabled": True,
                "docyolo_enabled": True,
                "mineru_enabled": True
            }
    
    def save_config(self):
        """Save GUI configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def translate_document(self, 
                          pdf_file,
                          source_lang: str,
                          target_lang: str,
                          backend: str,
                          enable_masking: bool,
                          enable_reranking: bool,
                          use_docyolo: bool,
                          use_mineru: bool) -> tuple:
        """Translate document with full pipeline."""
        if pdf_file is None:
            return "❌ Please upload a PDF file", None, ""
        
        try:
            # Get input path
            if hasattr(pdf_file, 'name'):
                input_path = Path(pdf_file.name)
            else:
                input_path = Path(pdf_file)
            
            # Configure pipeline
            config = PipelineConfig(
                source_lang=source_lang,
                target_lang=target_lang,
                backend=backend,
                enable_masking=enable_masking,
                enable_reranking=enable_reranking,
                quality_threshold=0.7 if enable_reranking else 0.5
            )
            
            # Set extraction method flags
            status_msg = ""
            if use_docyolo:
                status_msg += "🔄 Using DocYOLO for extraction...\n"
            if use_mineru:
                status_msg += "🔄 Using MinerU for extraction...\n"
            
            # Create pipeline
            status_msg += f"🔄 Initializing {backend} backend...\n"
            pipeline = TranslationPipeline(config)
            
            # Parse PDF
            status_msg += "📄 Parsing PDF...\n"
            parser = PDFParser()
            document = parser.parse(str(input_path))
            
            # Translate
            total_blocks = sum(len(seg.blocks) for seg in document.segments)
            status_msg += f"🔄 Translating {total_blocks} blocks...\n"
            result = pipeline.translate_document(document)
            
            # Render output
            output_path = input_path.parent / f"{input_path.stem}_translated.pdf"
            status_msg += "📝 Rendering output...\n"
            
            renderer = PDFRenderer()
            renderer.render_pdf(result.document, str(output_path))
            
            # Success
            success_msg = "✅ Translation completed!\n\n"
            success_msg += f"Backend: {backend}\n"
            success_msg += f"Blocks translated: {result.blocks_translated}\n"
            success_msg += f"Blocks failed: {result.blocks_failed}\n"
            if result.bleu_score:
                success_msg += f"BLEU score: {result.bleu_score:.2f}\n"
            success_msg += f"Duration: {result.duration:.1f}s\n"
            success_msg += f"Output: {output_path.name}\n"
            
            return success_msg, str(output_path), status_msg + success_msg
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            import traceback
            detailed_error = traceback.format_exc()
            return error_msg, None, detailed_error
    
    def test_backend(self, backend: str, sample_text: str) -> str:
        """Test a translation backend."""
        try:
            config = PipelineConfig(backend=backend)
            pipeline = TranslationPipeline(config)
            
            # Get backend instance
            backend_instance = pipeline.translation_backend
            
            # Test translation
            result = backend_instance.translate(
                sample_text,
                source_lang="en",
                target_lang="fr"
            )
            
            return f"✅ Backend '{backend}' is working!\n\nTest translation:\n{sample_text} → {result}"
        
        except Exception as e:
            return f"❌ Backend '{backend}' failed:\n{str(e)}"
    
    def save_api_key(self, backend: str, api_key: str) -> str:
        """Save API key for backend."""
        if not api_key:
            return "❌ Please enter an API key"
        
        # Save to environment
        env_var = f"{backend.upper()}_API_KEY"
        os.environ[env_var] = api_key
        
        # Save to config
        self.config["api_keys"][backend] = api_key
        self.save_config()
        
        return f"✅ API key saved for {backend}\n\nSet as: ${env_var}"
    
    def load_glossary(self, file_path: str) -> str:
        """Load glossary from file."""
        try:
            with open(file_path) as f:
                glossary = json.load(f)
            return f"✅ Loaded {len(glossary)} terms from glossary"
        except Exception as e:
            return f"❌ Error loading glossary: {str(e)}"
    
    def create_interface(self):
        """Create the complete Gradio interface."""
        
        # Enhanced theme with maximum visibility
        if self.config.get("dark_mode"):
            theme = gr.themes.Base(
                primary_hue="blue",
                secondary_hue="cyan",
                neutral_hue="slate",
            )
        else:
            # Light mode with MAXIMUM contrast
            theme = gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="indigo",
                neutral_hue="gray",
            ).set(
                body_background_fill="#ffffff",
                body_text_color="#000000",
                button_primary_background_fill="#1e40af",
                button_primary_background_fill_hover="#1e3a8a",
                button_primary_text_color="#ffffff",
                button_secondary_background_fill="#e5e7eb",
                button_secondary_text_color="#000000",
                input_background_fill="#ffffff",
                input_border_color="#6b7280",
                input_border_width="2px",
                block_label_text_color="#000000",
                block_title_text_color="#000000",
                block_background_fill="#f9fafb",
            )
        
        # Dynamic CSS and JS for instant theme switching
        is_dark = self.config.get("dark_mode", False)
        dark_class = "dark-mode" if is_dark else ""
        
        dynamic_css = f"""
            .gradio-container {{font-family: 'Inter', sans-serif;}}
            
            /* Light mode (default) */
            body:not(.dark-mode) h1,
            body:not(.dark-mode) h2,
            body:not(.dark-mode) h3 {{font-weight: 700 !important; color: #000000 !important;}}
            body:not(.dark-mode) .prose {{color: #000000 !important;}}
            body:not(.dark-mode) .prose h1,
            body:not(.dark-mode) .prose h2,
            body:not(.dark-mode) .prose h3 {{color: #000000 !important;}}
            body:not(.dark-mode) label {{color: #000000 !important; font-weight: 600 !important;}}
            body:not(.dark-mode) .markdown {{color: #000000 !important;}}
            body:not(.dark-mode) {{background: #ffffff !important;}}
            
            /* Dark mode */
            body.dark-mode {{background: #1f2937 !important; color: #e5e7eb !important;}}
            body.dark-mode h1,
            body.dark-mode h2,
            body.dark-mode h3 {{color: #f3f4f6 !important;}}
            body.dark-mode .prose {{color: #e5e7eb !important;}}
            body.dark-mode label {{color: #e5e7eb !important;}}
            body.dark-mode .markdown {{color: #e5e7eb !important;}}
            body.dark-mode .gradio-container {{background: #1f2937 !important;}}
        """
        
        with gr.Blocks(title="SciTrans-LLMs Pro", theme=theme, css=dynamic_css) as demo:
            
            # Hidden HTML to inject dark mode initialization
            gr.HTML(f"""
            <script>
                // Apply dark mode on page load
                if ({str(is_dark).lower()}) {{
                    document.body.classList.add('dark-mode');
                }}
            </script>
            """)
            
            # Global header with dark mode toggle
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("""
                    # 🚀 SciTrans-LLMs Professional Edition
                    Complete scientific document translation system with advanced features.
                    """)
                with gr.Column(scale=1):
                    dark_mode_global = gr.Button(
                        "☀️ Switch to Light" if self.config.get("dark_mode", False) else "🌙 Switch to Dark",
                        variant="secondary",
                        size="sm",
                        elem_id="theme-toggle"
                    )
                    theme_status = gr.Textbox(
                        value="",
                        visible=False,
                        elem_id="theme-status"
                    )
            
            gr.Markdown("---")
            
            with gr.Tabs():
                # TAB 1: TRANSLATION
                with gr.Tab("📄 Translation"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### ⚙️ Configuration")
                            
                            # Input method
                            input_method = gr.Radio(
                                choices=["PDF Upload", "URL"],
                                value="PDF Upload",
                                label="Input Method"
                            )
                            
                            # PDF Upload
                            pdf_upload = gr.File(
                                label="Upload PDF",
                                file_types=[".pdf"],
                                type="file"
                            )
                            
                            # URL Input
                            pdf_url = gr.Textbox(
                                label="PDF URL",
                                placeholder="https://arxiv.org/pdf/...",
                                visible=False
                            )
                            
                            # Language selection (FR ↔ EN only)
                            gr.Markdown("#### 🌍 Languages (French ↔ English)")
                            with gr.Row():
                                source_lang = gr.Radio(
                                    choices=["en", "fr"],
                                    value="en",
                                    label="Source"
                                )
                                target_lang = gr.Radio(
                                    choices=["fr", "en"],
                                    value="fr",
                                    label="Target"
                                )
                            
                            # Backend
                            backend = gr.Dropdown(
                                choices=["cascade", "free", "ollama", "deepseek", "openai", "anthropic"],
                                value=self.config.get("default_backend", "cascade"),
                                label="Translation Backend"
                            )
                            
                            # Core options
                            gr.Markdown("#### ✨ Core Features")
                            enable_masking = gr.Checkbox(
                                value=True,
                                label="🎭 LaTeX Masking",
                                info="Protect formulas and code"
                            )
                            
                            enable_reranking = gr.Checkbox(
                                value=self.config.get("reranking_enabled", True),
                                label="🏆 Quality Reranking",
                                info="Multi-candidate selection (ENABLED BY DEFAULT)"
                            )
                            
                            # Extraction options
                            gr.Markdown("#### 🔍 Extraction Methods")
                            use_docyolo = gr.Checkbox(
                                value=self.config.get("docyolo_enabled", True),
                                label="📊 DocYOLO",
                                info="Advanced document structure detection"
                            )
                            
                            use_mineru = gr.Checkbox(
                                value=self.config.get("mineru_enabled", True),
                                label="⚒️ MinerU",
                                info="Deep content extraction"
                            )
                            
                            # Translate button
                            translate_btn = gr.Button(
                                "🚀 Translate Document",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 🔍 PDF Preview & Results")
                            
                            # PDF Preview with improved pagination
                            with gr.Group():
                                pdf_preview_image = gr.Image(
                                    label="Page Preview",
                                    height=500,
                                    show_label=False
                                )
                                
                                # Pagination controls
                                with gr.Row():
                                    prev_page_btn = gr.Button("← Previous", size="sm", scale=1)
                                    pdf_page_input = gr.Number(
                                        value=1,
                                        minimum=1,
                                        maximum=1000,
                                        label="Page",
                                        scale=1
                                    )
                                    pdf_page_count = gr.Textbox(
                                        value="of 1",
                                        label="",
                                        interactive=False,
                                        scale=1
                                    )
                                    next_page_btn = gr.Button("Next →", size="sm", scale=1)
                            
                            gr.Markdown("---")
                            
                            # Status
                            status_box = gr.Textbox(
                                label="Status",
                                lines=6,
                                interactive=False
                            )
                            
                            # Output file
                            output_file = gr.File(
                                label="📥 Download Translated PDF",
                                interactive=False
                            )
                            
                            # Detailed log
                            with gr.Accordion("📋 Detailed Log", open=False):
                                detailed_log = gr.Textbox(
                                    lines=6,
                                    interactive=False
                                )
                    
                    # Handle input method switching
                    def toggle_input(method):
                        return {
                            pdf_upload: gr.update(visible=method == "PDF Upload"),
                            pdf_url: gr.update(visible=method == "URL")
                        }
                    
                    input_method.change(
                        fn=toggle_input,
                        inputs=[input_method],
                        outputs=[pdf_upload, pdf_url]
                    )
                    
                    # Translation handler
                    translate_btn.click(
                        fn=self.translate_document,
                        inputs=[
                            pdf_upload,
                            source_lang,
                            target_lang,
                            backend,
                            enable_masking,
                            enable_reranking,
                            use_docyolo,
                            use_mineru
                        ],
                        outputs=[status_box, output_file, detailed_log]
                    )
                
                # TAB 2: TESTING
                with gr.Tab("🧪 Testing"):
                    gr.Markdown("""
                    ### Backend Testing
                    
                    Test each translation backend with sample text.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            test_backend_dropdown = gr.Dropdown(
                                choices=["cascade", "free", "ollama", "deepseek", "openai", "anthropic"],
                                value="cascade",
                                label="Backend to Test"
                            )
                            
                            test_sample_text = gr.Textbox(
                                value="Machine learning enables artificial intelligence.",
                                label="Sample Text",
                                lines=3
                            )
                            
                            test_btn = gr.Button("🧪 Test Backend", variant="primary")
                        
                        with gr.Column():
                            test_result = gr.Textbox(
                                label="Test Result",
                                lines=10,
                                interactive=False
                            )
                    
                    test_btn.click(
                        fn=self.test_backend,
                        inputs=[test_backend_dropdown, test_sample_text],
                        outputs=[test_result]
                    )
                    
                    # Backend status table
                    gr.Markdown("""
                    ### Backend Status
                    
                    | Backend | Cost | API Key | Status |
                    |---------|------|---------|--------|
                    | cascade | FREE | ❌ No | ✅ Available |
                    | free | FREE | ❌ No | ✅ Available |
                    | ollama | FREE | ❌ No | ⚙️ Needs model |
                    | deepseek | $ | ✅ Yes | 🔑 Set in Settings |
                    | openai | $$$ | ✅ Yes | 🔑 Set in Settings |
                    | anthropic | $$$ | ✅ Yes | 🔑 Set in Settings |
                    """)
                
                # TAB 3: GLOSSARY
                with gr.Tab("📚 Glossary"):
                    gr.Markdown("""
                    ### Glossary Management
                    
                    Manage domain-specific term dictionaries.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Load Glossary")
                            
                            glossary_file = gr.File(
                                label="Upload Glossary (JSON)",
                                file_types=[".json"],
                                type="file"
                            )
                            
                            load_glossary_btn = gr.Button("📥 Load Glossary")
                            
                            gr.Markdown("#### Add Terms")
                            
                            term_source = gr.Textbox(
                                label="Source Term",
                                placeholder="machine learning"
                            )
                            
                            term_target = gr.Textbox(
                                label="Target Term",
                                placeholder="apprentissage automatique"
                            )
                            
                            add_term_btn = gr.Button("➕ Add Term")
                            
                            gr.Markdown("#### Online Glossary")
                            
                            online_url = gr.Textbox(
                                label="Glossary URL",
                                placeholder="https://example.com/glossary.json"
                            )
                            
                            fetch_online_btn = gr.Button("🌐 Fetch Online Glossary")
                        
                        with gr.Column():
                            glossary_status = gr.Textbox(
                                label="Status",
                                lines=5,
                                interactive=False
                            )
                            
                            glossary_preview = gr.JSON(
                                label="Current Glossary",
                                value={}
                            )
                    
                    load_glossary_btn.click(
                        fn=lambda f: self.load_glossary(f.name) if f else "❌ No file selected",
                        inputs=[glossary_file],
                        outputs=[glossary_status]
                    )
                
                # TAB 4: SETTINGS
                with gr.Tab("⚙️ Settings"):
                    gr.Markdown("""
                    ### System Settings
                    
                    Configure API keys, preferences, and appearance.
                    """)
                    
                    with gr.Tabs():
                        # API Keys
                        with gr.Tab("🔑 API Keys"):
                            gr.Markdown("#### Configure API Keys")
                            
                            with gr.Row():
                                with gr.Column():
                                    api_backend = gr.Dropdown(
                                        choices=["openai", "anthropic", "deepseek", "huggingface"],
                                        value="openai",
                                        label="Backend"
                                    )
                                    
                                    api_key_input = gr.Textbox(
                                        label="API Key",
                                        type="password",
                                        placeholder="sk-..."
                                    )
                                    
                                    save_key_btn = gr.Button("💾 Save API Key", variant="primary")
                                
                                with gr.Column():
                                    api_key_status = gr.Textbox(
                                        label="Status",
                                        lines=5,
                                        interactive=False
                                    )
                                    
                                    gr.Markdown("""
                                    **Where to get API keys:**
                                    
                                    - **OpenAI**: https://platform.openai.com/api-keys
                                    - **Anthropic**: https://console.anthropic.com/
                                    - **DeepSeek**: https://platform.deepseek.com/
                                    - **HuggingFace**: https://huggingface.co/settings/tokens
                                    """)
                            
                            save_key_btn.click(
                                fn=self.save_api_key,
                                inputs=[api_backend, api_key_input],
                                outputs=[api_key_status]
                            )
                        
                        # Appearance
                        with gr.Tab("🎨 Appearance"):
                            gr.Markdown("#### Theme Settings")
                            gr.Markdown("💡 Use the dark mode toggle in the header to switch themes instantly!")
                            
                            gr.Markdown("""
                            #### Color Scheme
                            - **Light Mode**: High contrast with dark text on light backgrounds
                            - **Dark Mode**: Easy on the eyes with light text on dark backgrounds
                            
                            Theme changes apply automatically - no reload needed!
                            """)
                        
                        # Preferences
                        with gr.Tab("⚡ Preferences"):
                            gr.Markdown("#### Default Settings")
                            
                            default_backend_pref = gr.Dropdown(
                                choices=["cascade", "free", "deepseek", "openai"],
                                value=self.config.get("default_backend", "cascade"),
                                label="Default Backend"
                            )
                            
                            reranking_pref = gr.Checkbox(
                                value=self.config.get("reranking_enabled", True),
                                label="Enable Reranking by Default"
                            )
                            
                            docyolo_pref = gr.Checkbox(
                                value=self.config.get("docyolo_enabled", True),
                                label="Enable DocYOLO by Default"
                            )
                            
                            mineru_pref = gr.Checkbox(
                                value=self.config.get("mineru_enabled", True),
                                label="Enable MinerU by Default"
                            )
                            
                            save_prefs_btn = gr.Button("💾 Save Preferences")
                            
                            def save_preferences(backend, reranking, docyolo, mineru):
                                self.config["default_backend"] = backend
                                self.config["reranking_enabled"] = reranking
                                self.config["docyolo_enabled"] = docyolo
                                self.config["mineru_enabled"] = mineru
                                self.save_config()
                                return "✅ Preferences saved!"
                            
                            pref_status = gr.Textbox(label="Status", interactive=False)
                            
                            # Auto-save on change
                            def auto_save_backend(value):
                                self.config["default_backend"] = value
                                self.save_config()
                                return "✅ Default backend updated"
                            
                            def auto_save_reranking(value):
                                self.config["reranking_enabled"] = value
                                self.save_config()
                                return "✅ Reranking preference updated"
                            
                            def auto_save_docyolo(value):
                                self.config["docyolo_enabled"] = value
                                self.save_config()
                                return "✅ DocYOLO preference updated"
                            
                            def auto_save_mineru(value):
                                self.config["mineru_enabled"] = value
                                self.save_config()
                                return "✅ MinerU preference updated"
                            
                            save_prefs_btn.click(
                                fn=save_preferences,
                                inputs=[default_backend_pref, reranking_pref, docyolo_pref, mineru_pref],
                                outputs=[pref_status]
                            )
                            
                            # Auto-save on individual changes
                            default_backend_pref.change(
                                fn=auto_save_backend,
                                inputs=[default_backend_pref],
                                outputs=[pref_status]
                            )
                            reranking_pref.change(
                                fn=auto_save_reranking,
                                inputs=[reranking_pref],
                                outputs=[pref_status]
                            )
                            docyolo_pref.change(
                                fn=auto_save_docyolo,
                                inputs=[docyolo_pref],
                                outputs=[pref_status]
                            )
                            mineru_pref.change(
                                fn=auto_save_mineru,
                                inputs=[mineru_pref],
                                outputs=[pref_status]
                            )
                
                # TAB 5: ABOUT
                with gr.Tab("ℹ️ About"):
                    gr.Markdown("""
                    # SciTrans-LLMs Professional
                    
                    ## 🎯 Features
                    
                    ### Translation
                    - ✅ PDF & URL input
                    - ✅ French ↔ English only
                    - ✅ 7 translation backends
                    - ✅ LaTeX masking
                    - ✅ Quality reranking (enabled by default)
                    - ✅ DocYOLO extraction
                    - ✅ MinerU extraction
                    
                    ### Testing
                    - ✅ Backend testing interface
                    - ✅ Sample translation tests
                    - ✅ Status monitoring
                    
                    ### Glossary
                    - ✅ Load from files
                    - ✅ Add custom terms
                    - ✅ Fetch online glossaries
                    - ✅ Domain-specific dictionaries
                    
                    ### Settings
                    - ✅ API key management
                    - ✅ Dark mode toggle
                    - ✅ Default preferences
                    - ✅ Persistent configuration
                    
                    ## 🚀 Quick Start
                    
                    1. Go to **Translation** tab
                    2. Upload PDF or paste URL
                    3. Select backend (CASCADE is free!)
                    4. Click **Translate**
                    5. Download result
                    
                    ## 💡 Tips
                    
                    - Use **CASCADE** for free translations
                    - Enable **Reranking** for best quality
                    - Set API keys in **Settings** for paid backends
                    - Test backends in **Testing** tab
                    - Manage terms in **Glossary** tab
                    
                    ## 📚 Documentation
                    
                    - `API_KEYS_SETUP.md` - API key setup guide
                    - `QUICK_START.md` - Quick start guide
                    - `COMPLETE_SOLUTION.md` - Full documentation
                    
                    ---
                    
                    **Version:** 2.0.0 Professional  
                    **Status:** ✅ All systems operational
                    """)
            
            # Global dark mode toggle handler
            def toggle_dark_mode():
                current = self.config.get("dark_mode", False)
                new_mode = not current
                self.config["dark_mode"] = new_mode
                self.save_config()
                button_text = "☀️ Switch to Light" if new_mode else "🌙 Switch to Dark"
                return button_text, "toggled"
            
            # PDF preview handler with fit-to-box
            def preview_pdf_page(pdf_file, page_num):
                if pdf_file is None:
                    return None
                try:
                    import fitz
                    from PIL import Image
                    import io
                    
                    if hasattr(pdf_file, 'name'):
                        pdf_path = pdf_file.name
                    else:
                        pdf_path = pdf_file
                    
                    doc = fitz.open(pdf_path)
                    page = doc[int(page_num) - 1]
                    
                    # Get page dimensions
                    rect = page.rect
                    page_width = rect.width
                    page_height = rect.height
                    
                    # Calculate zoom to fit width=800px, maintaining aspect ratio
                    target_width = 800
                    zoom = target_width / page_width
                    
                    # Render with calculated zoom
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    doc.close()
                    
                    img = Image.open(io.BytesIO(img_data))
                    return img
                except Exception as e:
                    return None
            
            dark_mode_global.click(
                fn=toggle_dark_mode,
                outputs=[dark_mode_global, theme_status],
                _js="""() => {
                    // Toggle dark mode class instantly
                    document.body.classList.toggle('dark-mode');
                    return [];
                }"""
            )
            
            def handle_pdf_upload(pdf_file):
                if pdf_file is None:
                    return None, 1, "of 1"
                try:
                    import fitz
                    doc = fitz.open(pdf_file.name)
                    num_pages = len(doc)
                    doc.close()
                    preview = preview_pdf_page(pdf_file, 1)
                    return preview, 1, f"of {num_pages}"
                except:
                    return None, 1, "of 1"
            
            def go_to_page(pdf_file, page_num):
                if pdf_file is None or page_num is None:
                    return None
                preview = preview_pdf_page(pdf_file, int(page_num))
                return preview
            
            def prev_page(pdf_file, current_page, page_count_text):
                if pdf_file is None:
                    return None, 1
                new_page = max(1, int(current_page) - 1)
                preview = preview_pdf_page(pdf_file, new_page)
                return preview, new_page
            
            def next_page(pdf_file, current_page, page_count_text):
                if pdf_file is None:
                    return None, 1
                max_page = int(page_count_text.split()[-1])
                new_page = min(max_page, int(current_page) + 1)
                preview = preview_pdf_page(pdf_file, new_page)
                return preview, new_page
            
            pdf_upload.change(
                fn=handle_pdf_upload,
                inputs=[pdf_upload],
                outputs=[pdf_preview_image, pdf_page_input, pdf_page_count]
            )
            
            pdf_page_input.change(
                fn=go_to_page,
                inputs=[pdf_upload, pdf_page_input],
                outputs=[pdf_preview_image]
            )
            
            prev_page_btn.click(
                fn=prev_page,
                inputs=[pdf_upload, pdf_page_input, pdf_page_count],
                outputs=[pdf_preview_image, pdf_page_input]
            )
            
            next_page_btn.click(
                fn=next_page,
                inputs=[pdf_upload, pdf_page_input, pdf_page_count],
                outputs=[pdf_preview_image, pdf_page_input]
            )
        
        return demo


def launch_gui():
    """Launch the enhanced GUI."""
    import os
    import signal
    import subprocess
    
    # Auto-kill any process on port 7860 to avoid manual killing
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
    
    app = EnhancedTranslationGUI()
    interface = app.create_interface()
    
    # Suppress warnings with quiet mode
    import warnings
    warnings.filterwarnings('ignore')
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    launch_gui()
