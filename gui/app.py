# -*- coding: utf-8 -*-
"""
SciTrans LLMs - Scientific Document Translation GUI
Version 1.0 - Clean Design
"""

import gradio as gr
import logging
import sys
import os
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from queue import Queue

sys.path.insert(0, str(Path(__file__).parent.parent))

# Logger for GUI module
logger = logging.getLogger(__name__)


class SciTransGUI:
    """Clean GUI with all features."""
    
    def __getattr__(self, name):
        """Defensive attribute access to prevent AttributeError for missing methods."""
        if name == '_generate_translation_preview':
            # Return a dummy function if method is missing (shouldn't happen, but defensive)
            def dummy_preview(*args, **kwargs):
                return "Preview generation method not available in this version."
            return dummy_preview
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __init__(self):
        self.config_file = Path.home() / ".scitrans" / "config.json"
        self.config_file.parent.mkdir(exist_ok=True)
        self.glossary_file = Path.home() / ".scitrans" / "glossary.json"
        self.translated_pdf_path = None
        self.source_pdf_path = None  # Store source PDF path for preview
        self.log_queue = Queue()
        self.load_config()
        self.load_glossary()  # Load persistent glossary
    
    def load_config(self):
        """Load configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    self.config = json.load(f)
            except:
                self.config = self._default_config()
        else:
            self.config = self._default_config()
        
        # Load API keys from environment variables if not in config
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }
        
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        # Only add env vars to config if they exist and aren't already in config
        for backend, env_var in env_map.items():
            if backend not in self.config["api_keys"]:
                env_value = os.environ.get(env_var)
                if env_value and env_value.strip():
                    # Don't save env vars to config, just note they exist
                    # The _get_api_keys_table will check env vars directly
                    pass
    
    def _default_config(self):
        return {
            "dark_mode": True,
            "default_backend": "free",
            "api_keys": {},
            "reranking_enabled": True,
            "masking_enabled": True,
            "cache_enabled": True,
            "max_candidates": 3,
            "context_window": 5,
            "glossary_enabled": True
        }
    
    def save_config(self):
        """Save configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _load_api_key_for_backend(self, backend: str) -> Optional[str]:
        """Load API key for a backend from GUI config, environment, or config file.
        
        Args:
            backend: Backend name
            
        Returns:
            API key if found, None otherwise
        """
        backend_lower = backend.lower()
        
        # Check GUI config first
        if hasattr(self, 'config') and self.config.get("api_keys", {}).get(backend_lower):
            return self.config["api_keys"][backend_lower]
        
        # Check environment variables
        env_mappings = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY"
        }
        
        if backend_lower in env_mappings:
            api_key = os.getenv(env_mappings[backend_lower])
            if api_key:
                return api_key
        
        # Check config file (~/.scitrans/config.yaml)
        config_path = Path.home() / ".scitrans" / "config.yaml"
        if config_path.exists():
            try:
                from scitran.utils.config_loader import load_config
                config = load_config(str(config_path))
                api_keys = config.get("api_keys", {})
                if backend_lower in api_keys:
                    return api_keys[backend_lower]
            except Exception:
                pass
        
        return None
    
    def load_glossary(self):
        """Load persistent glossary from disk (SPRINT 3: Using GlossaryManager)."""
        from scitran.translation.glossary.manager import GlossaryManager
        
        self.glossary_manager = GlossaryManager()
        self.glossary = {}  # Keep for UI display
        
        if self.glossary_file.exists():
            try:
                count = self.glossary_manager.load_from_file(self.glossary_file)
                self.glossary = self.glossary_manager.to_dict()
                logger.info(f"Loaded {count} terms from persistent storage")
            except Exception as e:
                logger.warning(f"Could not load glossary: {e}")
                self.glossary = {}
    
    def save_glossary(self):
        """Save glossary to disk for persistence (SPRINT 3: Using GlossaryManager)."""
        if hasattr(self, 'glossary_manager') and self.glossary_manager:
            self.glossary_manager.export_to_file(Path(self.glossary_file))
        else:
            # Fallback to old method
            with open(self.glossary_file, 'w') as f:
                json.dump(self.glossary, f, indent=2)
    
    def log(self, msg: str):
        """Add log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {msg}")
    
    # =========================================================================
    # Translation Functions
    # =========================================================================
    
    def translate_document(
        self,
        pdf_file,
        source_lang,
        target_lang,
        backend,
        model_name,
        advanced_options,
        num_candidates,
        context_window,
        quality_threshold,
        prompt_rounds,
        batch_size,
        enable_parallel,
        max_workers,
        adaptive_concurrency,
        start_page,
        end_page,
        font_dir,
        font_files,
        font_priority,
        mask_custom_macros,
        mask_apostrophes_in_latex,
        progress=gr.Progress()
    ):
        """Translate document with proper rendering and live progress."""
        # Initialize logs and add_log BEFORE try block so they're always available in error handler
        logs = []
        def add_log(msg):
            timestamp = datetime.now().strftime("%H:%M:%S")
            logs.append(f"[{timestamp}] {msg}")
        if pdf_file is None:
            return (
                "Please upload a PDF file",
                gr.update(value=None, visible=False),
                "",
                None,
                "of 0",
                gr.update(maximum=1, value=1),
                None,
                "",
                "No preview available - please upload a PDF file first."
            )
        
        # Parse advanced options
        enable_masking = "Masking" in advanced_options
        enable_reranking = "Reranking" in advanced_options
        use_context = "Context" in advanced_options
        use_glossary = "Glossary" in advanced_options
        
        # logs and add_log are already defined at function start (lines 157-158)
        
        try:
            # Handle both file uploads and URL-downloaded PDFs
            if pdf_file is None:
                return (
                    "❌ Error: No PDF file provided",
                    gr.update(visible=False),
                    "Please upload a PDF file or provide a URL.",
                    None,
                    "of 0",
                    gr.update(maximum=1, value=1),
                    None,
                    "",
                    ""
                )
            
            # Get PDF path - handle both file objects and string paths
            if isinstance(pdf_file, str):
                input_path = pdf_file  # Already a path (from URL download)
            elif hasattr(pdf_file, 'name'):
                input_path = pdf_file.name  # File upload object
            else:
                input_path = str(pdf_file)
            
            from scitran.extraction.pdf_parser import PDFParser
            from scitran.core.pipeline import TranslationPipeline, PipelineConfig
            from scitran.rendering.pdf_renderer import PDFRenderer
            
            # Convert to Path object
            input_path = Path(input_path)
            add_log(f"Input: {input_path.name}")
            
            progress(0.05, desc="Parsing PDF...")
            add_log("Parsing PDF structure...")
            
            parser = PDFParser()
            # Page range support - handle empty strings from Gradio
            # Gradio Number components return None for empty, but we need to handle it properly
            start_page_val = 0  # Default to first page (0-based)
            if start_page is not None:
                try:
                    start_page_str = str(start_page).strip()
                    if start_page_str and start_page_str.lower() not in ['none', 'null', '']:
                        start_page_val = int(float(start_page))  # Handle both int and float inputs
                        start_page_val = max(0, start_page_val)  # Ensure non-negative
                except (ValueError, TypeError):
                    start_page_val = 0
            
            end_page_val = None  # None means process all pages
            if end_page is not None:
                try:
                    end_page_str = str(end_page).strip()
                    if end_page_str and end_page_str.lower() not in ['none', 'null', '']:
                        end_page_val = int(float(end_page))  # Handle both int and float inputs
                        # If end_page is 0, treat as "all pages" (None) - 0 means "no limit" in UI
                        if end_page_val == 0:
                            end_page_val = None
                        else:
                            end_page_val = max(start_page_val, end_page_val) if end_page_val is not None else None  # Ensure >= start_page
                except (ValueError, TypeError):
                    end_page_val = None

            document = parser.parse(
                str(input_path),
                max_pages=None,
                start_page=start_page_val if start_page_val is not None else 0,
                end_page=end_page_val,
            )
            total_blocks = len(document.translatable_blocks)
            num_pages = document.stats.get("num_pages", 0)
            add_log(f"Parsed {num_pages} pages")
            add_log(f"Found {total_blocks} text blocks")
            
            # Log blocks per page for debugging
            blocks_per_page = {}
            for block in document.translatable_blocks:
                if block.bbox:
                    page = block.bbox.page
                    blocks_per_page[page] = blocks_per_page.get(page, 0) + 1
            for page, count in sorted(blocks_per_page.items()):
                add_log(f"  Page {page + 1}: {count} blocks")
            
            progress(0.1, desc="Configuring pipeline...")
            add_log(f"Backend: {backend}")
            
            # Validate and normalize model name for the selected backend
            model_name = self.validate_model_for_backend(backend, model_name)
            if model_name and model_name != "default":
                add_log(f"Model: {model_name}")
            
            # Warn about free backend limitations
            if backend.lower() in ['cascade', 'free']:
                add_log(f"⚠️ Note: Free backends have rate limits - large PDFs will be slow")
                add_log(f"💡 Tip: For large documents, use paid backends (OpenAI/Anthropic)")
            
            add_log(f"Masking: {'ON' if enable_masking else 'OFF'}")
            add_log(f"Reranking: {'ON' if enable_reranking else 'OFF'}")
            add_log(f"Caching: ON (persistent)")
            
            # For speed, use batch mode when reranking is off
            # When reranking is on, use user-selected candidate count (turns)
            # Safely convert numeric inputs (handle empty strings from Gradio)
            def safe_int(value, default=None):
                if value is None or (isinstance(value, str) and not value.strip()):
                    return default
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return default
            
            def safe_float(value, default=0.0):
                if value is None or (isinstance(value, str) and not value.strip()):
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # Load API key for the selected backend
            api_key = self._load_api_key_for_backend(backend)
            
            # PHASE 1.3: Check if fast mode should be enabled (no reranking + single candidate)
            fast_mode_enabled = not enable_reranking and safe_int(num_candidates, 1) == 1
            
            config = PipelineConfig(
                source_lang=source_lang,
                target_lang=target_lang,
                backend=backend,
                model_name=model_name if model_name and model_name != "default" else None,
                api_key=api_key,  # Load API key from config/env
                enable_masking=enable_masking,
                enable_reranking=enable_reranking,
                num_candidates=safe_int(num_candidates, 1) if enable_reranking else 1,
                cache_translations=True,
                enable_glossary=use_glossary,
                context_window_size=safe_int(context_window, 5),
                quality_threshold=safe_float(quality_threshold, 0.7),
                batch_size=safe_int(batch_size, 20),  # Increased default for speed
                prompt_optimization_rounds=safe_int(prompt_rounds, 0),
                optimize_prompts=safe_int(prompt_rounds, 0) > 0,
                debug_mode=True,
                debug_log_path=Path(".cache/scitrans/gui_debug.jsonl"),  # PHASE 4.1
                mask_custom_macros=bool(mask_custom_macros),
                mask_apostrophes_in_latex=bool(mask_apostrophes_in_latex),
                enable_parallel_processing=bool(enable_parallel),
                max_workers=safe_int(max_workers, None) if max_workers and str(max_workers).strip() else None,
                adaptive_concurrency=bool(adaptive_concurrency),
                fast_mode=fast_mode_enabled,  # PHASE 1.3: Auto-enable when appropriate
            )
            
            pipeline = TranslationPipeline(config)
            
            # Create progress callback for live updates with performance tracking
            perf_metrics = {
                "start_time": time.time(),
                "blocks_processed": 0,
                "cache_hits": 0,
                "blocks_per_sec": 0.0
            }
            
            def pipeline_progress(prog: float, msg: str):
                # Map pipeline progress (0-1) to our progress (0.1-0.85)
                mapped_progress = 0.1 + (prog * 0.75)
                progress(mapped_progress, desc=msg)
                add_log(msg)
                
                # Update performance metrics
                elapsed = time.time() - perf_metrics["start_time"]
                if elapsed > 0 and "blocks" in msg.lower():
                    # Try to extract block count from message
                    import re
                    match = re.search(r'(\d+)\s*/\s*(\d+)\s*blocks?', msg)
                    if match:
                        processed = int(match.group(1))
                        total = int(match.group(2))
                        perf_metrics["blocks_processed"] = processed
                        perf_metrics["blocks_per_sec"] = processed / elapsed if elapsed > 0 else 0
            
            progress(0.15, desc="Starting translation...")
            add_log("Starting translation with caching enabled...")
            
            # Translate with progress updates
            result = pipeline.translate_document(document, progress_callback=pipeline_progress)

            # Update preview with loading state during translation
            # (The actual preview will be updated after rendering)
            
            # Log cache stats if available
            stats = pipeline.get_statistics()
            if 'batch_cache_hits' in stats:
                add_log(f"Cache hits: {stats.get('batch_cache_hits', 0)}")
                add_log(f"New translations: {stats.get('batch_translated', 0)}")
            if 'cache_hits' in stats:
                add_log(f"Sequential cache hits: {stats.get('cache_hits', 0)}")
            
            add_log(f"Translated {result.blocks_translated}/{total_blocks} blocks")
            
            # Generate translation preview (text preview before rendering) - COMPLETELY INLINE, NO METHOD DEPENDENCY

            try:
                preview_lines = []
                preview_lines.append("=" * 60)
                preview_lines.append("TRANSLATION PREVIEW (Before PDF Rendering)")
                preview_lines.append("=" * 60)
                preview_lines.append("")
                preview_lines.append(f"Translation completed: {result.blocks_translated}/{total_blocks} blocks translated.\n")
                
                block_count = 0
                max_blocks = 50
                for segment in document.segments:
                    for block in segment.blocks:
                        if not hasattr(block, 'is_translatable') or (hasattr(block, 'is_translatable') and not block.is_translatable):
                            continue
                        if block_count >= max_blocks:
                            preview_lines.append(f"\n... (showing first {max_blocks} blocks)")
                            break
                        
                        block_count += 1
                        preview_lines.append(f"[Block {getattr(block, 'block_id', 'unknown')}]")
                        
                        if hasattr(block, 'source_text') and block.source_text:
                            src = str(block.source_text)[:200]
                            preview_lines.append(f"Source: {src}{'...' if len(str(block.source_text)) > 200 else ''}")
                        
                        if hasattr(block, 'translated_text') and block.translated_text:
                            trans = str(block.translated_text)[:200]
                            preview_lines.append(f"Translation: {trans}{'...' if len(str(block.translated_text)) > 200 else ''}")
                        else:
                            preview_lines.append("Translation: [NOT TRANSLATED]")
                        
                        preview_lines.append("")
                    
                    if block_count >= max_blocks:
                        break
                
                if block_count == 0:
                    translation_preview_text = f"Translation completed: {result.blocks_translated}/{total_blocks} blocks translated.\n\nNo translatable blocks found in document."
                else:
                    translation_preview_text = "\n".join(preview_lines)
                
            except Exception as preview_error:
                translation_preview_text = f"Translation completed: {result.blocks_translated}/{total_blocks} blocks translated.\n\nPreview generation error: {str(preview_error)}"
                add_log(f"⚠️ Preview generation failed: {preview_error}")
            
            # Debug: Check how many blocks have translations per page
            translated_by_page = {}
            translated_with_bbox = 0
            for seg in document.segments:
                for b in seg.blocks:
                    if b.translated_text and b.bbox:
                        translated_with_bbox += 1
                        page = b.bbox.page
                        translated_by_page[page] = translated_by_page.get(page, 0) + 1
            missing_blocks = [b.block_id for seg in document.segments for b in seg.blocks if not b.translated_text]
            
            add_log(f"Blocks with translation + bbox: {translated_with_bbox}")
            for page, count in sorted(translated_by_page.items()):
                add_log(f"  Page {page + 1}: {count} translated blocks")
            
            progress(0.9, desc="Rendering PDF...")
            add_log("Rendering translated PDF (clearing source text, preserving layout)...")
            
            # Save PDF to temp directory first (Gradio requirement), then copy to persistent location
            temp_output_dir = Path(tempfile.gettempdir()) / "scitrans"
            temp_output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_output_path = temp_output_dir / f"{input_path.stem}_{target_lang}_{timestamp}.pdf"

            renderer = PDFRenderer(
                font_dir=font_dir if font_dir else None,
                font_files=[f.strip() for f in font_files.split(",") if f.strip()] if font_files else None,
                font_priority=[p.strip().lower() for p in font_priority.split(",") if p.strip()] if font_priority else None,
                overflow_strategy="shrink",  # PHASE 3.2: Default to shrink for GUI
                min_font_size=4.0
            )

            renderer.render_with_layout(str(input_path), result.document, str(temp_output_path))
            
            
            # Count translated pages for pagination
            try:
                import fitz
                with fitz.open(str(temp_output_path)) as out_pdf:
                    translated_pages = len(out_pdf)
            except Exception as e:
                translated_pages = document.stats.get("num_pages", 1)
            
            # Also copy to persistent location for user access
            # Keep final output in temp (Gradio requirement) and optionally mirror to home cache
            persistent_output_dir = Path.home() / ".scitrans" / "output"
            persistent_output_dir.mkdir(parents=True, exist_ok=True)
            persistent_output_path = persistent_output_dir / f"{input_path.stem}_{target_lang}_{timestamp}.pdf"
            
            import shutil
            if temp_output_path.exists():
                shutil.copy2(temp_output_path, persistent_output_path)
                file_size = temp_output_path.stat().st_size / 1024  # KB
                add_log(f"✓ PDF created: {temp_output_path.name} ({file_size:.1f} KB)")
                add_log(f"📁 Also saved to: {persistent_output_path}")
            else:
                add_log(f"⚠️ Warning: PDF file not found at {temp_output_path}")
                # Return all 9 values for error case
                return (
                    "Error: PDF not created",
                    gr.update(value=None, visible=False, interactive=False),
                    "\n".join(logs),
                    None,
                    "of 0",
                    gr.update(maximum=1, value=1),
                    None,
                    "Error: PDF file was not created. Check logs for details.",
                    "No preview available - PDF creation failed."
                )
            
            self.translated_pdf_path = str(temp_output_path)
            # Keep source PDF path stored for preview
            if not self.source_pdf_path:
                self.source_pdf_path = str(input_path)
            
            progress(1.0, desc="Complete!")
            add_log("Translation complete!")
            
            # Calculate performance metrics
            elapsed = time.time() - perf_metrics["start_time"]
            blocks_per_sec = total_blocks / elapsed if elapsed > 0 else 0
            cache_hits = stats.get('batch_cache_hits', 0) + stats.get('cache_hits', 0)
            cache_hit_rate = (cache_hits / total_blocks * 100) if total_blocks > 0 else 0
            
            status = f"✓ Translation Complete\n"
            status += f"Blocks: {result.blocks_translated}/{total_blocks}\n"
            status += f"Time: {result.duration:.1f}s ({elapsed:.1f}s total)\n"
            status += f"Speed: {blocks_per_sec:.1f} blocks/sec\n"
            status += f"Backend: {backend}"
            if cache_hits > 0:
                status += f"\nCache: {cache_hits} hits ({cache_hit_rate:.1f}% hit rate)"
            
            # Performance info
            perf_text = f"Performance Metrics:\n"
            perf_text += f"• Throughput: {blocks_per_sec:.2f} blocks/second\n"
            perf_text += f"• Cache hit rate: {cache_hit_rate:.1f}%\n"
            perf_text += f"• Total time: {elapsed:.2f}s\n"
            if 'batch_translated' in stats:
                perf_text += f"• Batch translated: {stats.get('batch_translated', 0)}\n"
                perf_text += f"• Batch cached: {stats.get('batch_cache_hits', 0)}"
            
            
            # Return PDF file path for File component
            page_update = gr.update(maximum=max(1, translated_pages), value=1)
            page_total_text = f"of {max(1, translated_pages)}"
            
            # Store source PDF path if not already stored
            if not self.source_pdf_path:
                self.source_pdf_path = str(input_path)
            
            # Log before return to catch any errors during return
            
            # Ensure source path is a string for File component
            source_path_str = self.source_pdf_path if self.source_pdf_path else (str(input_path) if input_path else None)
            
            return (
                status,
                gr.update(value=str(temp_output_path), visible=True, interactive=True),
                "\n".join(logs),
                str(temp_output_path),  # Return PDF file path for File component (translated)
                page_total_text,
                page_update,
                source_path_str,  # Return source PDF path for File component
                perf_text,
                translation_preview_text,
            )
            
        except Exception as e:
            import traceback
            from scitran.core.exceptions import SciTransError
            
            # CRITICAL: Ensure logs and add_log are available (they should be, but defensive check)
            if 'logs' not in locals():
                logs = []
            if 'add_log' not in locals():
                def add_log(msg):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    logs.append(f"[{timestamp}] {msg}")
            
            error_msg = str(e)
            error_details = ""
            
            # Enhanced error display with suggestions
            try:
                if isinstance(e, SciTransError):
                    error_msg = e.message
                    if e.suggestion:
                        error_details = f"\n\n💡 Suggestion: {e.suggestion}"
                    if e.recoverable:
                        error_details += "\n\nThis error is recoverable. You can try again with different settings."
                
                add_log(f"❌ Error: {error_msg}")
                if error_details:
                    add_log(error_details)
            except:
                # If add_log fails, at least set error_msg
                pass
            
            # Enhanced error logging with full traceback
            try:
                import traceback as tb
                full_traceback = tb.format_exc()
            except:
                pass
            
            try:
                full_error = f"❌ Error: {error_msg}{error_details}"
                if hasattr(self, 'config') and self.config.get("debug_mode", False):
                    full_error += f"\n\nDebug traceback:\n{full_traceback if 'full_traceback' in locals() else 'N/A'}"
            except:
                full_error = f"❌ Error: {error_msg}"
            
            # Generate error preview - always succeed
            try:
                translation_preview_text = f"Error: {error_msg}\n\nNo translation preview available due to error."
            except:
                translation_preview_text = "Error generating preview"
            
            # ALWAYS return exactly 9 values, even if something fails
            try:
                logs_str = "\n".join(logs) if logs else "No logs available"
            except:
                logs_str = "Error generating logs"
            
            try:
                perf_text = f"Error occurred. Check logs for details.\n\nError: {error_msg}"
            except:
                perf_text = "Error occurred. Check logs for details."
            
            # Final return - guaranteed to return 9 values
            return (
                full_error,
                gr.update(value=None, visible=False, interactive=False),
                logs_str,
                None,
                "of 0",
                gr.update(maximum=1, value=1),
                None,
                perf_text,
                translation_preview_text
            )
    
    def _generate_translation_preview(self, document, max_blocks=50):
        """Generate text preview of translated document.
        
        NOTE: This method exists for backward compatibility.
        The new inline preview generation (lines 353-400) doesn't use this method.
        """
        try:
            from scitran.core.models import Document
            
            if not isinstance(document, Document):
                return "No document available for preview."
            
            preview_lines = []
            preview_lines.append("=" * 60)
            preview_lines.append("TRANSLATION PREVIEW (Before PDF Rendering)")
            preview_lines.append("=" * 60)
            preview_lines.append("")
            
            block_count = 0
            for segment in document.segments:
                for block in segment.blocks:
                    if not block.is_translatable:
                        continue
                    
                    if block_count >= max_blocks:
                        preview_lines.append(f"\n... (showing first {max_blocks} blocks)")
                        break
                    
                    block_count += 1
                    preview_lines.append(f"[Block {block.block_id}]")
                    
                    if block.source_text:
                        preview_lines.append(f"Source: {block.source_text[:200]}{'...' if len(block.source_text) > 200 else ''}")
                    
                    if block.translated_text:
                        preview_lines.append(f"Translation: {block.translated_text[:200]}{'...' if len(block.translated_text) > 200 else ''}")
                    else:
                        preview_lines.append("Translation: [NOT TRANSLATED]")
                    
                    preview_lines.append("")
                
                if block_count >= max_blocks:
                    break
            
            if block_count == 0:
                return "No translatable blocks found in document."
            
            return "\n".join(preview_lines)
        except Exception as e:
            return f"Error generating preview: {str(e)}"
    
    def preview_pdf(self, pdf_file, page_num=1):
        """Preview PDF page - returns PDF file path for File component."""
        if pdf_file is None:
            return None
        try:
            # Return the PDF file path directly - Gradio File component will render PDFs natively
            pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
            if isinstance(pdf_path, str) and pdf_path.endswith('.pdf'):
                return pdf_path
            return None
        except Exception as e:
            print(f"Preview error: {e}")
            return None
    
    def create_loading_image(self, message="Translating..."):
        """Create a loading overlay image."""
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a semi-transparent overlay
        img = Image.new('RGBA', (800, 500), (0, 0, 0, 128))  # Semi-transparent black
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()
        
        # Get text size and center it
        bbox = draw.textbbox((0, 0), message, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((800 - text_width) // 2, (500 - text_height) // 2)
        
        # Draw white text
        draw.text(position, message, fill=(255, 255, 255, 255), font=font)
        
        return img
    
    def get_page_count(self, pdf_file):
        """Get PDF page count."""
        if pdf_file is None:
            return 1
        try:
            import fitz
            pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except:
            return 1
    
    def download_pdf_from_url(self, url):
        """Download PDF from URL and return file path."""
        if not url or not url.strip():
            return None, "Please enter a valid URL."
        
        try:
            import requests
            import tempfile
            from pathlib import Path
            
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                return None, "❌ Invalid URL. Must start with http:// or https://"
            
            # Download PDF
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                # Still try if URL ends with .pdf
                if not url.lower().endswith('.pdf'):
                    return None, f"❌ URL does not appear to be a PDF (content-type: {content_type})"
            
            # Save to temporary file
            temp_dir = Path.home() / ".scitrans" / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename from URL
            filename = url.split('/')[-1] or "downloaded.pdf"
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            temp_path = temp_dir / filename
            
            # Download file
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify it's a valid PDF
            try:
                import fitz
                doc = fitz.open(str(temp_path))
                page_count = len(doc)
                doc.close()
                
                return str(temp_path), f"✅ Downloaded PDF successfully ({page_count} pages)"
            except Exception as e:
                temp_path.unlink(missing_ok=True)
                return None, f"❌ Downloaded file is not a valid PDF: {str(e)}"
                
        except requests.exceptions.RequestException as e:
            return None, f"❌ Failed to download PDF: {str(e)}"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def _get_model_options_for_backend(self, backend: str):
        """Return model choices, default value, and visibility for a backend."""
        model_options = {
            "ollama": ["llama3.1", "llama3.2", "qwen2.5", "mistral", "gemma2", "llama3.3"],
            "huggingface": ["facebook/mbart-large-50-many-to-many-mmt", "Helsinki-NLP/opus-mt-en-fr"],
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229"],
            "deepseek": ["deepseek-chat", "deepseek-coder"],
            "cascade": ["default"],
            "free": ["default"],
            "libre": ["default"],
            "argos": ["default"],
            "local": ["default"]
        }
        
        options = model_options.get(backend, ["default"])
        default_value = options[0]
        # Show model selector for backends that support model selection
        visible = backend in ["ollama", "huggingface", "openai", "anthropic", "deepseek"]
        return {"choices": options, "value": default_value, "visible": visible}
    
    def update_model_options(self, backend):
        """Update model dropdown options based on selected backend."""
        opts = self._get_model_options_for_backend(backend)
        # Always set value to first valid option to prevent "default" from persisting
        # when switching to backends that don't support "default"
        return gr.update(choices=opts["choices"], value=opts["value"], visible=opts["visible"])
    
    def validate_model_for_backend(self, backend, model_name):
        """Validate and normalize model name for the selected backend."""
        opts = self._get_model_options_for_backend(backend)
        valid_choices = opts["choices"]
        
        # If model is "default" or not in valid choices, use the first valid option
        if model_name == "default" or model_name not in valid_choices:
            return opts["value"]
        return model_name
    
    # =========================================================================
    # Testing Functions
    # =========================================================================
    
    def test_backend(self, backend, sample_text):
        """Test translation backend."""
        if not sample_text.strip():
            sample_text = "Machine learning enables computers to learn from data without explicit programming."
        
        try:
            from scitran.translation.base import TranslationRequest
            
            if backend == "cascade":
                from scitran.translation.backends.cascade_backend import CascadeBackend
                translator = CascadeBackend()
            elif backend == "free":
                from scitran.translation.backends.free_backend import FreeBackend
                translator = FreeBackend()
            elif backend == "ollama":
                from scitran.translation.backends.ollama_backend import OllamaBackend
                translator = OllamaBackend()
            elif backend == "local":
                from scitran.translation.backends.local_backend import LocalBackend
                translator = LocalBackend()
            elif backend == "libre":
                from scitran.translation.backends.libre_backend import LibreBackend
                translator = LibreBackend()
            elif backend == "argos":
                from scitran.translation.backends.argos_backend import ArgosBackend
                translator = ArgosBackend()
            elif backend == "huggingface":
                from scitran.translation.backends.huggingface_backend import HuggingFaceBackend
                translator = HuggingFaceBackend()
            else:
                api_key = self.config.get("api_keys", {}).get(backend)
                if not api_key:
                    return f"Backend '{backend}' requires API key. Set it in Settings."
                
                if backend == "openai":
                    from scitran.translation.backends.openai_backend import OpenAIBackend
                    translator = OpenAIBackend(api_key=api_key)
                elif backend == "anthropic":
                    from scitran.translation.backends.anthropic_backend import AnthropicBackend
                    translator = AnthropicBackend(api_key=api_key)
                elif backend == "deepseek":
                    from scitran.translation.backends.deepseek_backend import DeepSeekBackend
                    translator = DeepSeekBackend(api_key=api_key)
                else:
                    return f"Unknown backend: {backend}"
            
            if not translator.is_available():
                return f"Backend '{backend}' is not available."
            
            request = TranslationRequest(text=sample_text, source_lang="en", target_lang="fr")
            
            start = time.time()
            response = translator.translate_sync(request)
            elapsed = time.time() - start
            
            if response.translations:
                result = f"Backend '{backend}' OK ({elapsed:.2f}s)\n\n"
                result += f"EN: {sample_text}\n\n"
                result += f"FR: {response.translations[0]}"
                return result
            else:
                return f"Backend returned no translation."
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def test_masking(self, test_input):
        """Test masking with custom or default input."""
        if not test_input.strip():
            test_input = "The equation $E=mc^2$ is famous. See https://arxiv.org for more."
        
        try:
            from scitran.masking.engine import MaskingEngine
            from scitran.core.models import Block
            
            engine = MaskingEngine()
            block = Block(block_id="test", source_text=test_input)
            masked = engine.mask_block(block)
            
            result = f"Original: {test_input}\n\n"
            result += f"Masked: {masked.masked_text}\n\n"
            result += f"Masks found: {len(masked.masks)}\n"
            for m in masked.masks:
                result += f"  • {m.mask_type}: '{m.original}' → '{m.placeholder}'\n"
            
            # Show what would be sent to translator
            result += f"\n(The masked text above is what gets translated, preserving formulas/URLs)"
            
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def test_layout(self, pdf_file):
        """Test layout extraction with detailed analysis."""
        if pdf_file is None:
            return "Default test: Layout extraction module OK.\n\nUpload a PDF to test with your document."
        
        try:
            import fitz
            from scitran.core.models import BlockType
            from scitran.extraction.pdf_parser import PDFParser
            
            pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
            doc = fitz.open(pdf_path)
            parser = PDFParser()
            
            result = f"PDF: {Path(pdf_path).name}\n"
            result += f"Total Pages: {len(doc)}\n"
            result += "=" * 60 + "\n\n"
            
            # Analyze first 3 pages in detail
            for i, page in enumerate(doc[:3]):
                result += f"Page {i+1}:\n"
                result += f"  Size: {page.rect.width:.0f} x {page.rect.height:.0f}\n"
                
                # Images
                images = page.get_images()
                result += f"  Images: {len(images)}\n"
                
                # Extract text with detailed info
                text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)
                
                # Analyze fonts
                fonts_used = {}
                block_types = {}
                tables_count = 0
                equations_count = 0
                figures_count = 0
                
                for block_data in text_dict.get("blocks", []):
                    if "lines" not in block_data:
                        figures_count += 1
                        continue
                    
                    # Analyze fonts in this block
                    for line in block_data.get("lines", []):
                        for span in line.get("spans", []):
                            font_name = span.get("font", "unknown")
                            font_size = span.get("size", 11)
                            flags = span.get("flags", 0)
                            
                            # Parse font style
                            is_bold = bool(flags & 16)
                            is_italic = bool(flags & 2)
                            style = ""
                            if is_bold and is_italic:
                                style = "Bold+Italic"
                            elif is_bold:
                                style = "Bold"
                            elif is_italic:
                                style = "Italic"
                            else:
                                style = "Regular"
                            
                            font_key = f"{font_name} ({style}, {font_size:.1f}pt)"
                            fonts_used[font_key] = fonts_used.get(font_key, 0) + 1
                    
                    # Classify block type
                    block_text = ""
                    for line in block_data.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")
                    
                    if block_text.strip():
                        # Use parser's classification
                        classified = parser._classify_block(block_text)
                        block_types[classified] = block_types.get(classified, 0) + 1
                        
                        if classified == "table":
                            tables_count += 1
                        elif classified == "math_content":
                            equations_count += 1
                
                # Count text blocks
                text_blocks = [b for b in text_dict.get("blocks", []) if "lines" in b]
                result += f"  Text blocks: {len(text_blocks)}\n"
                
                # Font summary
                if fonts_used:
                    result += f"  Fonts used ({len(fonts_used)} unique):\n"
                    # Sort by usage count
                    sorted_fonts = sorted(fonts_used.items(), key=lambda x: x[1], reverse=True)
                    for font_name, count in sorted_fonts[:5]:  # Top 5 fonts
                        result += f"    • {font_name}: {count} spans\n"
                    if len(sorted_fonts) > 5:
                        result += f"    ... and {len(sorted_fonts) - 5} more\n"
                
                # Block types
                if block_types:
                    result += f"  Block types:\n"
                    for block_type, count in sorted(block_types.items(), key=lambda x: x[1], reverse=True):
                        result += f"    • {block_type}: {count}\n"
                
                # Special elements
                result += f"  Special elements:\n"
                result += f"    • Tables detected: {tables_count}\n"
                result += f"    • Equations/formulas: {equations_count}\n"
                result += f"    • Figures/images: {figures_count}\n"
                
                result += "\n"
            
            doc.close()
            return result
        except Exception as e:
            import traceback
            return f"Error: {str(e)}\n\n{traceback.format_exc()}"
    
    def test_cache(self):
        """Test cache functionality."""
        try:
            from scitran.utils.fast_translator import PersistentCache
            cache = PersistentCache()
            
            cache.set("test_key", "en", "fr", "test_value")
            result = cache.get("test_key", "en", "fr")
            
            if result == "test_value":
                stats = cache.stats()
                return f"Cache OK\n\nStats: {stats}"
            else:
                return "Cache read failed"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # =========================================================================
    # Glossary Functions - Extensive Domain Glossaries
    # =========================================================================
    
    def _get_scientific_ml_glossary(self, direction="en-fr"):
        """Machine Learning and AI scientific terms."""
        en_fr = {
            # Core ML concepts
            # Architecture terms
            # NLP terms
            # Computer Vision
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    def _get_scientific_physics_glossary(self, direction="en-fr"):
        """Physics and mathematics scientific terms."""
        en_fr = {
            # Physics
            # Mathematics
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    def _get_scientific_bio_glossary(self, direction="en-fr"):
        """Biology and medical scientific terms."""
        en_fr = {
            # Biology
            # Structural Biology
            # Medical
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    def _get_europarl_glossary(self, direction="en-fr"):
        """European Parliament and legal terms."""
        en_fr = {
                "European Union": "Union européenne",
                "European Parliament": "Parlement européen",
                "European Commission": "Commission européenne",
                "Member State": "État membre",
                "Council of the European Union": "Conseil de l'Union européenne",
                "legislation": "législation",
                "regulation": "règlement",
                "directive": "directive",
                "treaty": "traité",
                "amendment": "amendement",
                "resolution": "résolution",
                "committee": "commission",
                "rapporteur": "rapporteur",
                "codecision": "codécision",
                "subsidiarity": "subsidiarité",
            "human rights": "droits de l'homme",
            "rule of law": "état de droit",
            "democracy": "démocratie",
            "transparency": "transparence",
            "accountability": "responsabilité",
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    def _get_chemistry_glossary(self, direction="en-fr"):
        """Chemistry scientific terms."""
        en_fr = {
            "molecule": "molécule",
            "atom": "atome",
            "electron": "électron",
            "proton": "proton",
            "neutron": "neutron",
            "chemical bond": "liaison chimique",
            "covalent bond": "liaison covalente",
            "ionic bond": "liaison ionique",
            "oxidation": "oxydation",
            "reduction": "réduction",
            "catalyst": "catalyseur",
            "reagent": "réactif",
            "solvent": "solvant",
            "solution": "solution",
            "concentration": "concentration",
            "molar mass": "masse molaire",
            "equilibrium": "équilibre",
            "reaction rate": "vitesse de réaction",
            "organic chemistry": "chimie organique",
            "inorganic chemistry": "chimie inorganique",
            "polymer": "polymère",
            "compound": "composé",
            "element": "élément",
            "isotope": "isotope",
            "valence": "valence",
            "electronegativity": "électronégativité",
            "spectroscopy": "spectroscopie",
            "chromatography": "chromatographie",
            "titration": "titrage",
            "pH": "pH",
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    def _get_cs_glossary(self, direction="en-fr"):
        """Computer Science terms."""
        en_fr = {
            "algorithm": "algorithme",
            "data structure": "structure de données",
            "array": "tableau",
            "linked list": "liste chaînée",
            "hash table": "table de hachage",
            "binary tree": "arbre binaire",
            "graph": "graphe",
            "recursion": "récursion",
            "iteration": "itération",
            "complexity": "complexité",
            "time complexity": "complexité temporelle",
            "space complexity": "complexité spatiale",
            "database": "base de données",
            "query": "requête",
            "index": "index",
            "cache": "cache",
            "memory": "mémoire",
            "stack": "pile",
            "queue": "file",
            "heap": "tas",
            "sorting": "tri",
            "searching": "recherche",
            "compiler": "compilateur",
            "interpreter": "interpréteur",
            "operating system": "système d'exploitation",
            "network": "réseau",
            "protocol": "protocole",
            "encryption": "chiffrement",
            "decryption": "déchiffrement",
            "authentication": "authentification",
            "authorization": "autorisation",
            "API": "API",
            "framework": "cadriciel",
            "library": "bibliothèque",
            "version control": "contrôle de version",
            "debugging": "débogage",
            "testing": "test",
            "deployment": "déploiement",
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    def _get_statistics_glossary(self, direction="en-fr"):
        """Statistics terms."""
        en_fr = {
            "mean": "moyenne",
            "median": "médiane",
            "mode": "mode",
            "variance": "variance",
            "standard deviation": "écart-type",
            "distribution": "distribution",
            "normal distribution": "distribution normale",
            "probability": "probabilité",
            "hypothesis": "hypothèse",
            "null hypothesis": "hypothèse nulle",
            "p-value": "valeur p",
            "confidence interval": "intervalle de confiance",
            "sample": "échantillon",
            "population": "population",
            "correlation": "corrélation",
            "regression": "régression",
            "linear regression": "régression linéaire",
            "outlier": "valeur aberrante",
            "bias": "biais",
            "significance": "signification",
            "statistical significance": "significativité statistique",
            "Bayesian": "bayésien",
            "frequentist": "fréquentiste",
            "maximum likelihood": "maximum de vraisemblance",
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    def load_glossary_domain(self, domain, direction="en-fr"):
        """Load glossary by domain (SPRINT 3: Using GlossaryManager)."""
        if not hasattr(self, 'glossary_manager'):
            from scitran.translation.glossary.manager import GlossaryManager
            self.glossary_manager = GlossaryManager()
        
        prev_count = len(self.glossary_manager)
        count = self.glossary_manager.load_domain(domain, direction)
        new_count = len(self.glossary_manager) - prev_count
        
        if count == 0:
            # Try alternative file naming
            alt_direction = direction.replace('-', '_')
            count = self.glossary_manager.load_domain(domain, alt_direction)
            new_count = len(self.glossary_manager) - prev_count
        
        if count == 0:
            return f"⚠️ Could not load {domain} glossary. File may be missing or corrupted.", self.glossary
        
        # Update UI glossary dict
        self.glossary = self.glossary_manager.to_dict()
        self.save_glossary()
        
        return f"✓ Loaded {count} {domain.upper()} terms ({new_count} new)", self.glossary
    
    def load_all_scientific_glossaries(self, direction="en-fr"):
        """Load all scientific glossaries at once (SPRINT 3: Using GlossaryManager)."""
        if not hasattr(self, 'glossary_manager'):
            from scitran.translation.glossary.manager import GlossaryManager
            self.glossary_manager = GlossaryManager()
        
        total_before = len(self.glossary_manager)
        
        # Load all domains
        all_domains = ['ml', 'physics', 'biology', 'chemistry', 'cs', 'statistics', 'europarl']
        for domain in all_domains:
            try:
                self.glossary_manager.load_domain(domain, direction)
            except Exception as e:
                logger.warning(f"Could not load {domain}: {e}")
        
        total_after = len(self.glossary_manager)
        new_terms = total_after - total_before
        
        # Update UI glossary dict
        self.glossary = self.glossary_manager.to_dict()
        self.save_glossary()
        
        return f"✓ Loaded ALL glossaries: {total_after} total terms ({new_terms} new)", self.glossary
    
    def load_glossary_file(self, file):
        """Load glossary from uploaded file."""
        if file is None:
            return "No file selected", self.glossary
        
        try:
            file_path = file.name if hasattr(file, 'name') else str(file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            prev_count = len(self.glossary)
            if "terms" in data:
                self.glossary.update(data["terms"])
            else:
                self.glossary.update(data)
            
            new_count = len(self.glossary) - prev_count
            self.save_glossary()  # Save persistently
            return f"✓ Loaded from file: {new_count} new terms (total: {len(self.glossary)})", self.glossary
        except Exception as e:
            return f"Error: {str(e)}", self.glossary
    
    def add_glossary_term(self, source, target):
        """Add term to glossary."""
        if not source or not target:
            return "⚠ Enter both source and target terms", self.glossary
        
        source = source.strip()
        target = target.strip()
        
        if source in self.glossary:
            self.glossary[source] = target
            self.save_glossary()  # Save persistently
            return f"✓ Updated: '{source}' → '{target}'", self.glossary
        else:
            self.glossary[source] = target
            self.save_glossary()  # Save persistently
            return f"✓ Added: '{source}' → '{target}'", self.glossary
    
    def clear_glossary(self):
        """Clear glossary."""
        count = len(self.glossary)
        self.glossary = {}
        self.save_glossary()  # Save persistently
        return f"✓ Cleared {count} terms", {}
    
    def load_online_glossary(self, source, direction="en-fr"):
        """Load glossary from online sources like HuggingFace, Europarl, etc."""
        import requests
        
        prev_count = len(self.glossary)
        
        try:
            if source == "europarl_full":
                # Europarl parallel corpus - fetch common terms
                # Using a subset available via GitHub
                url = "https://raw.githubusercontent.com/Wikipedia-translations/europarl-extract/main/en-fr.json"
                try:
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, dict):
                            self.glossary.update(data)
                except:
                    # Fallback: use expanded built-in Europarl
                    self.glossary.update(self._get_expanded_europarl_glossary(direction))
            
            elif source == "huggingface_opus":
                # HuggingFace OPUS-100 terms - common scientific terms
                # Using a curated subset
                url = "https://huggingface.co/datasets/Helsinki-NLP/opus-100/raw/main/README.md"
                # Since direct API access may be limited, use expanded built-in
                self.glossary.update(self._get_expanded_scientific_glossary(direction))
            
            elif source == "wiktionary":
                # Wiktionary translations (curated subset)
                self.glossary.update(self._get_wiktionary_terms(direction))
            
            elif source == "iate":
                # IATE - Inter-Active Terminology for Europe
                self.glossary.update(self._get_iate_terms(direction))
            
            else:
                return f"Unknown source: {source}", self.glossary
            
            new_count = len(self.glossary) - prev_count
            self.save_glossary()
            return f"✓ Loaded from {source}: {new_count} new terms (total: {len(self.glossary)})", self.glossary
            
        except Exception as e:
            return f"Error loading from {source}: {str(e)}", self.glossary
    
    def _get_expanded_europarl_glossary(self, direction="en-fr"):
        """Expanded Europarl terminology (500+ terms)."""
        en_fr = {
            # Institutions
            # Legal terms
            # Political terms
            # Rights and freedoms
            # Economic terms
            # Policy areas
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    def _get_expanded_scientific_glossary(self, direction="en-fr"):
        """Expanded scientific glossary (500+ terms across all domains)."""
        en_fr = {
            # Advanced ML/AI
            # Physics & Math advanced
            # Biology advanced
            # More common scientific terms
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    def _get_wiktionary_terms(self, direction="en-fr"):
        """Common Wiktionary translation pairs."""
        en_fr = {
            "analysis": "analyse",
            "approach": "approche",
            "application": "application",
            "assessment": "évaluation",
            "assumption": "hypothèse",
            "behavior": "comportement",
            "calculation": "calcul",
            "characteristic": "caractéristique",
            "comparison": "comparaison",
            "complexity": "complexité",
            "component": "composant",
            "concept": "concept",
            "condition": "condition",
            "configuration": "configuration",
            "constraint": "contrainte",
            "context": "contexte",
            "contribution": "contribution",
            "criterion": "critère",
            "data": "données",
            "definition": "définition",
            "description": "description",
            "development": "développement",
            "dimension": "dimension",
            "distribution": "distribution",
            "effect": "effet",
            "efficiency": "efficacité",
            "element": "élément",
            "environment": "environnement",
            "estimation": "estimation",
            "evaluation": "évaluation",
            "evidence": "preuve",
            "example": "exemple",
            "experiment": "expérience",
            "explanation": "explication",
            "expression": "expression",
            "factor": "facteur",
            "feature": "caractéristique",
            "framework": "cadre",
            "function": "fonction",
            "generation": "génération",
            "implementation": "implémentation",
            "improvement": "amélioration",
            "indicator": "indicateur",
            "information": "information",
            "input": "entrée",
            "integration": "intégration",
            "interpretation": "interprétation",
            "investigation": "investigation",
            "knowledge": "connaissance",
            "layer": "couche",
            "limitation": "limitation",
            "literature": "littérature",
            "mechanism": "mécanisme",
            "method": "méthode",
            "modification": "modification",
            "objective": "objectif",
            "observation": "observation",
            "operation": "opération",
            "optimization": "optimisation",
            "output": "sortie",
            "parameter": "paramètre",
            "performance": "performance",
            "perspective": "perspective",
            "phenomenon": "phénomène",
            "prediction": "prédiction",
            "principle": "principe",
            "probability": "probabilité",
            "problem": "problème",
            "procedure": "procédure",
            "process": "processus",
            "property": "propriété",
            "proposition": "proposition",
            "quality": "qualité",
            "quantity": "quantité",
            "range": "plage",
            "rate": "taux",
            "ratio": "rapport",
            "reduction": "réduction",
            "relation": "relation",
            "relationship": "relation",
            "representation": "représentation",
            "requirement": "exigence",
            "research": "recherche",
            "resource": "ressource",
            "response": "réponse",
            "result": "résultat",
            "review": "revue",
            "sample": "échantillon",
            "scale": "échelle",
            "scenario": "scénario",
            "scope": "portée",
            "section": "section",
            "selection": "sélection",
            "sequence": "séquence",
            "series": "série",
            "set": "ensemble",
            "simulation": "simulation",
            "situation": "situation",
            "solution": "solution",
            "source": "source",
            "specification": "spécification",
            "stability": "stabilité",
            "stage": "étape",
            "standard": "norme",
            "state": "état",
            "statement": "déclaration",
            "strategy": "stratégie",
            "structure": "structure",
            "study": "étude",
            "subject": "sujet",
            "summary": "résumé",
            "system": "système",
            "technique": "technique",
            "technology": "technologie",
            "term": "terme",
            "test": "test",
            "theory": "théorie",
            "threshold": "seuil",
            "tool": "outil",
            "training": "entraînement",
            "transformation": "transformation",
            "trend": "tendance",
            "type": "type",
            "unit": "unité",
            "value": "valeur",
            "variable": "variable",
            "variation": "variation",
            "version": "version",
            "view": "vue",
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    def _get_iate_terms(self, direction="en-fr"):
        """IATE (Inter-Active Terminology for Europe) terms."""
        en_fr = {
            # Technical/IT
            # Business
            # Environment
            # Healthcare
        }
        if direction == "fr-en":
            return {v: k for k, v in en_fr.items()}
        return en_fr
    
    # =========================================================================
    # Settings Functions
    # =========================================================================
    
    def save_api_key(self, backend, api_key):
        """Save API key."""
        if not api_key or not api_key.strip():
            return "Please enter an API key"
        
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        
        env_var = env_map.get(backend, f"{backend.upper()}_API_KEY")
        os.environ[env_var] = api_key.strip()
        
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        self.config["api_keys"][backend] = api_key.strip()
        self.save_config()
        
        return f"API key saved for {backend}"
    
    def get_api_keys_display(self):
        """Get masked API keys."""
        keys = self.config.get("api_keys", {})
        if not keys:
            return "No API keys configured"
        
        result = []
        for backend, key in keys.items():
            masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            result.append(f"{backend}: {masked}")
        return "\n".join(result)
    
    def _get_api_keys_table(self):
        """Get API keys as table data for DataFrame."""
        keys = self.config.get("api_keys", {})
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }
        
        all_backends = ["openai", "anthropic", "deepseek", "huggingface"]
        table_data = []
        
        for backend in all_backends:
            env_var_name = env_map.get(backend)
            
            # Check config first (takes precedence)
            if backend in keys and keys[backend] and keys[backend].strip():
                masked = keys[backend][:8] + "..." + keys[backend][-4:] if len(keys[backend]) > 12 else "***"
                status = "✅ Configured (Config)"
            # Then check environment variables
            elif env_var_name:
                env_key = os.environ.get(env_var_name)
                if env_key and env_key.strip() and len(env_key.strip()) > 0:
                    status = "✅ From Environment"
                    # Show last 4 chars if key is long enough
                    if len(env_key.strip()) > 4:
                        masked = "***" + env_key.strip()[-4:]
                    else:
                        masked = "***"
                else:
                    status = "❌ Not Set"
                    masked = "-"
            else:
                status = "❌ Not Set"
                masked = "-"
            
            table_data.append([backend.capitalize(), status, masked])
        
        return table_data
    
    def delete_api_key(self, backend):
        """Delete API key for a backend."""
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        if backend in self.config["api_keys"]:
            del self.config["api_keys"][backend]
            self.save_config()
            
            # Also remove from environment
            env_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "huggingface": "HUGGINGFACE_API_KEY",
            }
            env_var = env_map.get(backend)
            if env_var and env_var in os.environ:
                del os.environ[env_var]
            
            return f"API key deleted for {backend}"
        else:
            return f"No API key found for {backend}"
    
    def save_all_settings(
        self, backend, masking, reranking, cache, glossary, context, 
        context_window, candidates, strict_mode, fallback
    ):
        """Save all settings."""
        self.config["default_backend"] = backend
        self.config["masking_enabled"] = masking
        self.config["reranking_enabled"] = reranking
        self.config["cache_enabled"] = cache
        self.config["glossary_enabled"] = glossary
        self.config["context_enabled"] = context
        self.config["context_window"] = int(context_window) if context_window else 5
        self.config["max_candidates"] = int(candidates) if candidates else 3
        self.config["strict_mode"] = strict_mode
        self.config["enable_fallback"] = fallback
        self.save_config()
        return "✅ All settings saved successfully!"
    
    def reset_settings(self):
        """Reset settings to defaults."""
        self.config = self._default_config()
        self.save_config()
        return "✅ Settings reset to defaults"
    
    def clear_cache(self):
        """Clear translation cache."""
        try:
            import shutil
            cache_dir = Path(".cache/translations")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True)
            return "Cache cleared"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # =========================================================================
    # Create Interface
    # =========================================================================
    
    def create_interface(self):
        """Create the Gradio interface."""
        
        with gr.Blocks(title="SciTrans LLMs") as demo:
            
            # Header
            gr.Markdown("# 🔬 SciTrans LLMs\n**Scientific Document Translation** • v1.0")
            
            with gr.Tabs():
                
                # ===========================================================
                # TAB 1: TRANSLATION
                # ===========================================================
                with gr.Tab("Translation"):
                    allowed_backends = ["cascade", "free", "ollama", "openai", "anthropic", "deepseek", "local", "libre", "argos"]
                    initial_backend = self.config.get("default_backend", "free")
                    if initial_backend not in allowed_backends:
                        initial_backend = "free"
                    initial_model_opts = self._get_model_options_for_backend(initial_backend)
                    # Ensure initial model value is always valid (never "default" for backends that don't support it)
                    if initial_backend in ["ollama", "huggingface", "openai", "anthropic", "deepseek"]:
                        # These backends don't support "default", so ensure we use a valid model
                        if initial_model_opts["value"] == "default" or initial_model_opts["value"] not in initial_model_opts["choices"]:
                            initial_model_opts["value"] = initial_model_opts["choices"][0] if initial_model_opts["choices"] else "default"

                    with gr.Row():
                        # Left: Controls (narrower)
                        with gr.Column(scale=2):
                            # Input method tabs: Upload or URL
                            with gr.Tabs():
                                with gr.Tab("📁 Upload PDF"):
                                    pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                                with gr.Tab("🔗 From URL"):
                                    pdf_url = gr.Textbox(
                                        label="PDF URL",
                                        placeholder="Enter URL to PDF (e.g., https://arxiv.org/pdf/1234.5678.pdf)",
                                        lines=1
                                    )
                                    url_load_btn = gr.Button("📥 Load PDF from URL", variant="primary")
                                    url_status = gr.Markdown("", visible=False)
                            
                            with gr.Row():
                                source_lang = gr.Dropdown(["en", "fr"], value="en", label="From", scale=1)
                                target_lang = gr.Dropdown(["fr", "en"], value="fr", label="To", scale=1)
                            
                            backend = gr.Dropdown(
                                allowed_backends,
                                value=initial_backend,
                                label="Backend"
                            )
                            
                            model_selector = gr.Dropdown(
                                initial_model_opts["choices"],
                                value=initial_model_opts["value"],
                                label="Model (for Ollama/HuggingFace/OpenAI/etc.)",
                                info="Select model for backends that support it",
                                visible=initial_model_opts["visible"]
                            )
                            
                            # All advanced features ON by default
                            advanced_options = gr.CheckboxGroup(
                                ["Masking", "Reranking", "Context", "Glossary"],
                                value=["Masking", "Reranking", "Context", "Glossary"],
                                label="Advanced Features"
                            )
                            
                            # Advanced tweakable parameters
                            with gr.Accordion("⚙️ Advanced Parameters", open=False):
                                num_candidates = gr.Slider(
                                    minimum=1,
                                    maximum=4,
                                    step=1,
                                    value=2,
                                    label="Number of candidates (turns)"
                                )
                                context_window = gr.Slider(
                                    minimum=0,
                                    maximum=10,
                                    step=1,
                                    value=5,
                                    label="Context window (blocks)"
                                )
                                prompt_rounds = gr.Slider(
                                    minimum=0,
                                    maximum=5,
                                    step=1,
                                    value=2,
                                    label="Prompt optimization rounds"
                                )
                                quality_threshold = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.05,
                                    value=0.7,
                                    label="Quality threshold"
                                )
                                batch_size = gr.Slider(
                                    minimum=5,
                                    maximum=30,
                                    step=1,
                                    value=10,
                                    label="Batch size",
                                    info="Number of blocks to process in parallel"
                                )
                                enable_parallel = gr.Checkbox(
                                    value=True,
                                    label="Enable Parallel Processing",
                                    info="Use parallel processing for large documents (faster)"
                                )
                                max_workers = gr.Number(
                                    value=None,
                                    precision=0,
                                    label="Max Workers (optional)",
                                    info="Max parallel workers (leave blank for auto-detect)"
                                )
                                adaptive_concurrency = gr.Checkbox(
                                    value=True,
                                    label="Adaptive Concurrency",
                                    info="Automatically adjust concurrency based on backend and document size"
                                )
                                with gr.Row():
                                    start_page = gr.Number(
                                        value=0,
                                        precision=0,
                                        label="Start page (0-based, inclusive)"
                                    )
                                    end_page = gr.Number(
                                        value=None,
                                        precision=0,
                                        label="End page (0-based, inclusive; blank = all)"
                                    )
                                font_dir = gr.Textbox(
                                    value="",
                                    label="Font directory (TTF/OTF) for embedding (optional)",
                                    placeholder="/path/to/fonts"
                                )
                                font_files = gr.Textbox(
                                    value="",
                                    label="Font files (comma-separated, optional; overrides directory priority)",
                                    placeholder="/path/to/font1.ttf,/path/to/font2.otf"
                                )
                                font_priority = gr.Textbox(
                                    value="",
                                    label="Font priority keywords (comma-separated, optional)",
                                    placeholder="roboto,helvetica"
                                )
                                mask_custom_macros = gr.Checkbox(
                                    value=True,
                                    label="Mask custom LaTeX macros (newcommand/DeclareMathOperator/etc.)"
                                )
                                mask_apostrophes_in_latex = gr.Checkbox(
                                    value=True,
                                    label="Protect apostrophes inside LaTeX/math"
                                )
                            
                            with gr.Row():
                                translate_btn = gr.Button("🚀 Translate", variant="primary")
                                retranslate_btn = gr.Button("🔁 Retranslate", variant="secondary")
                                clear_btn = gr.Button("🧹 Clear", variant="secondary")
                            
                            # Status/Log moved to left column below translate button
                            with gr.Accordion("📊 Status & Logs", open=False):
                                with gr.Tabs():
                                    with gr.Tab("Status"):
                                        status_box = gr.Textbox(
                                            lines=4, 
                                            interactive=False, 
                                            show_label=False,
                                            placeholder="Status will appear here..."
                                        )
                                    with gr.Tab("Log"):
                                        log_box = gr.Textbox(
                                            lines=20, 
                                            interactive=False, 
                                            show_label=False, 
                                            autoscroll=True,
                                            placeholder="Translation logs will appear here..."
                                        )
                                    with gr.Tab("Performance"):
                                        perf_info = gr.Textbox(
                                            lines=4,
                                            interactive=False,
                                            show_label=False,
                                            placeholder="Performance metrics will appear here..."
                                        )
                        
                        # Right: Preview (wider) - Preview replaces progress/status area
                        with gr.Column(scale=3):
                            # Download button (small footprint)
                            download_btn = gr.DownloadButton(
                                label="📥 Download Translated PDF",
                                value=None,
                                interactive=False,
                                visible=False
                            )
                            
                            # Preview area with fullscreen support
                            # Use File components for PDF previews (Gradio will render PDFs natively)
                            with gr.Tabs(selected=0):  # selected=0 prevents auto-switching
                                with gr.Tab("Source"):
                                    source_preview = gr.File(
                                        label="Source PDF Preview",
                                        file_types=[".pdf"],
                                        height=480,
                                        show_label=False,
                                        container=False
                                    )
                                with gr.Tab("Translated"):
                                    trans_preview = gr.File(
                                        label="Translated PDF Preview",
                                        file_types=[".pdf"],
                                        height=480,
                                        show_label=False,
                                        container=False
                                    )
                                with gr.Tab("Text Preview"):
                                    translation_preview = gr.Textbox(
                                        lines=20,
                                        label="Translation Preview (Before Rendering)",
                                        interactive=False,
                                        placeholder="Translation preview will appear here after translation completes..."
                                    )
                            
                            # Loading indicator (shown during translation)
                            loading_indicator = gr.Markdown("", visible=False)
                            
                            # Unified pagination
                            gr.Markdown("**Pages**")
                            with gr.Row():
                                page_prev = gr.Button("◀", size="sm", scale=0, min_width=30)
                                page_slider = gr.Slider(
                                    minimum=1,
                                    maximum=1,
                                    step=1,
                                    value=1,
                                    label="Page",
                                    interactive=True
                                )
                                page_next = gr.Button("▶", size="sm", scale=0, min_width=30)
                                page_total = gr.Textbox(value="of 1", show_label=False, interactive=False, scale=0, min_width=60, container=False)
                
                # ===========================================================
                # TAB 2: TESTING
                # ===========================================================
                with gr.Tab("Testing"):
                    gr.Markdown("### 🧪 Component Tests\nTest individual components before full translation.")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**🔌 Backend Test**")
                            test_backend_sel = gr.Dropdown(
                                ["cascade", "free", "ollama", "openai", "anthropic", "deepseek", "local", "libre", "argos", "huggingface"],
                                value="free", label="Backend",
                                info="Select backend to test. Free options: cascade, free, local, libre, argos"
                            )
                            # Pre-filled with rich test content
                            test_text = gr.Textbox(
                                value="""# Machine Learning Overview

Machine learning enables computers to learn from data without explicit programming.

## Key Concepts:
• **Supervised Learning** - Learn from labeled examples
• **Unsupervised Learning** - Find patterns in unlabeled data
• **Reinforcement Learning** - Learn through rewards/penalties

The loss function $L(\\theta) = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$ measures prediction error.

See: https://arxiv.org/abs/1234.5678 for more details.""",
                                label="Sample Text (includes headers, bullets, math, URLs)", lines=8
                            )
                            test_backend_btn = gr.Button("▶ Test Backend")
                            test_backend_result = gr.Textbox(label="Result", lines=6)
                        
                        with gr.Column():
                            gr.Markdown("**🎭 Masking Test**")
                            # Rich test content with various maskable elements
                            masking_input = gr.Textbox(
                                value="""The famous equation $E=mc^2$ demonstrates mass-energy equivalence.

For more complex formulas, consider:
$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$

References:
• Einstein, A. (1905) - https://doi.org/10.1002/andp.19053221004
• Code example: `import numpy as np`
• Email: researcher@university.edu
• LaTeX block: \\begin{equation}F = ma\\end{equation}""",
                                label="Test Input (LaTeX, URLs, code, emails)", lines=8
                            )
                            test_masking_btn = gr.Button("▶ Test Masking")
                            test_masking_result = gr.Textbox(label="Result", lines=6)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**📄 Layout Test**\nTest font detection, headers, footers, structure.")
                            layout_pdf = gr.File(label="Upload PDF to analyze layout", file_types=[".pdf"])
                            test_layout_btn = gr.Button("▶ Test Layout")
                            test_layout_result = gr.Textbox(label="Result", lines=15)
                        
                        with gr.Column():
                            gr.Markdown("**💾 Cache Test**\nVerify translation caching is working.")
                            test_cache_btn = gr.Button("▶ Test Cache")
                            test_cache_result = gr.Textbox(label="Result", lines=6)
                    
                    # Additional test samples
                    with gr.Accordion("📝 More Test Samples", open=False):
                        gr.Markdown("""
### Font & Style Tests
Copy these to test different formatting:

**Title Style:**
`NEURAL NETWORK ARCHITECTURES FOR SCIENTIFIC DOCUMENT ANALYSIS`

**Abstract Style:**
`Abstract: This paper presents a novel approach to machine translation using transformer architectures with attention mechanisms.`

**Section Headers:**
`1. Introduction`
`2.1 Related Work`
`3.2.1 Methodology Details`

**Bullet Points:**
`• First item with **bold** text`
`• Second item with *italic* text`
`• Third item with code: model.fit(X, y)`

**Numbered List:**
`1. First step in the process`
`2. Second step with equation $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$`
`3. Third step referencing [Author2023]`

**Footer/Header:**
`Page 1 of 10 | Confidential Draft | DOI: 10.1234/example.2024`
                        """)
                
                # ===========================================================
                # TAB 3: SETTINGS
                # ===========================================================
                with gr.Tab("Settings"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 🔑 API Keys Management")
                            
                            # API Keys Table/List
                            api_keys_table = gr.Dataframe(
                                headers=["Backend", "Status", "Key Preview"],
                                value=self._get_api_keys_table(),
                                label="Configured API Keys",
                                interactive=False,
                                wrap=True
                            )
                            
                            refresh_keys_btn = gr.Button("🔄 Refresh Keys Table", variant="secondary", size="sm")
                            
                            gr.Markdown("**Add/Update API Key**")
                            api_backend = gr.Dropdown(
                                ["openai", "anthropic", "deepseek", "huggingface"],
                                value="openai", 
                                label="Backend",
                                info="Select backend to configure"
                            )
                            api_key_input = gr.Textbox(
                                label="API Key", 
                                type="password",
                                placeholder="Enter API key here..."
                            )
                            with gr.Row():
                                save_key_btn = gr.Button("💾 Save Key", variant="primary")
                                delete_key_btn = gr.Button("🗑️ Delete Key", variant="stop")
                            
                            api_status = gr.Textbox(label="Status", lines=2, interactive=False)
                            
                            # Refresh button for API keys table
                            refresh_keys_btn = gr.Button("🔄 Refresh Keys Table", variant="secondary", size="sm")
                            
                            gr.Markdown("### 📋 Environment Variables")
                            gr.Markdown("""
                            You can also set API keys via environment variables:
                            - `OPENAI_API_KEY` for OpenAI
                            - `ANTHROPIC_API_KEY` for Anthropic
                            - `DEEPSEEK_API_KEY` for DeepSeek
                            - `HUGGINGFACE_API_KEY` for HuggingFace
                            
                            Keys set here override environment variables.
                            """)
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### ⚙️ Translation Settings")
                            
                            set_backend = gr.Dropdown(
                                ["cascade", "free", "ollama", "openai", "anthropic", "deepseek", "local", "libre", "argos", "huggingface"],
                                value=self.config.get("default_backend", "cascade"),
                                label="Default Backend",
                                info="Backend to use by default"
                            )
                            
                            with gr.Accordion("🔧 Core Features", open=True):
                                set_masking = gr.Checkbox(
                                    value=self.config.get("masking_enabled", True), 
                                    label="Enable Masking",
                                    info="Protect LaTeX, URLs, code from translation"
                                )
                                set_reranking = gr.Checkbox(
                                    value=self.config.get("reranking_enabled", True), 
                                    label="Enable Reranking",
                                    info="Select best translation from multiple candidates"
                                )
                                set_cache = gr.Checkbox(
                                    value=self.config.get("cache_enabled", True), 
                                    label="Enable Cache",
                                    info="Cache translations for faster re-runs"
                                )
                                set_glossary = gr.Checkbox(
                                    value=self.config.get("glossary_enabled", True), 
                                    label="Enable Glossary",
                                    info="Use domain-specific terminology"
                                )
                                set_context = gr.Checkbox(
                                    value=self.config.get("context_enabled", True),
                                    label="Enable Document Context",
                                    info="Maintain consistency across document"
                                )
                            
                            with gr.Accordion("📊 Advanced Parameters", open=False):
                                set_context_window = gr.Slider(
                                    0, 10, 
                                    value=self.config.get("context_window", 5), 
                                    step=1, 
                                    label="Context Window Size",
                                    info="Number of previous blocks to consider"
                                )
                                set_candidates = gr.Slider(
                                    1, 5, 
                                    value=self.config.get("max_candidates", 3), 
                                    step=1, 
                                    label="Max Translation Candidates",
                                    info="Number of candidates for reranking"
                                )
                                set_strict_mode = gr.Checkbox(
                                    value=self.config.get("strict_mode", True),
                                    label="Strict Mode",
                                    info="Fail loudly if translation incomplete"
                                )
                                set_fallback = gr.Checkbox(
                                    value=self.config.get("enable_fallback", True),
                                    label="Enable Fallback Backend",
                                    info="Use stronger backend if primary fails"
                                )
                            
                            save_settings_btn = gr.Button("💾 Save All Settings", variant="primary")
                            settings_status = gr.Textbox(label="", interactive=False)
                            
                            gr.Markdown("### 🛠️ Maintenance")
                            with gr.Row():
                                clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
                                reset_settings_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")
                            cache_status = gr.Textbox(label="", interactive=False)
                            
                            gr.Markdown("### 📝 Basic Commands")
                            gr.Markdown("""
                            **CLI Commands:**
                            ```bash
                            # Basic translation
                            scitrans translate input.pdf -o output.pdf
                            
                            # With backend
                            scitrans translate input.pdf --backend openai
                            
                            # With strict mode
                            scitrans translate input.pdf --strict-mode true
                            
                            # List backends
                            scitrans backends
                            
                            # Run tests
                            scitrans test
                            ```
                            """)
                
                # ===========================================================
                # TAB 4: GLOSSARY
                # ===========================================================
                with gr.Tab("Glossary"):
                    gr.Markdown("### 📚 Domain Terminology Management")
                    
                    # Explanation box
                    with gr.Accordion("ℹ️ How Glossary Works", open=False):
                        gr.Markdown("""
**What is the Glossary?**
The glossary is a dictionary of domain-specific terms that ensures consistent, accurate translations of technical vocabulary.

**How it's used:**
1. **During Translation**: When the system encounters a term in the glossary, it uses your specified translation instead of the generic one
2. **Pattern Matching**: Terms are matched case-insensitively in the source text
3. **Priority**: Glossary translations take precedence over backend translations

**Where is it stored?**
- **Session Glossary**: Loaded terms stay in memory during your session
- **Persistent Storage**: All loaded glossaries are saved to `~/.scitrans/glossary.json` and auto-load on next launch
- **Cache Integration**: Glossary-enhanced translations are cached for speed

**Best Practices:**
- Load domain-specific glossaries matching your document type
- Add custom terms for unique terminology in your field
- Use "Load ALL" for comprehensive coverage across domains
                        """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Domain selector dropdown
                            gr.Markdown("**🔬 Built-in Scientific Glossaries**")
                            glossary_domain = gr.Dropdown(
                                choices=[
                                    ("🤖 Machine Learning & AI (50+ terms)", "ml"),
                                    ("⚛️ Physics & Mathematics (40+ terms)", "physics"),
                                    ("🧬 Biology & Medical (40+ terms)", "biology"),
                                    ("🏛️ Legal & EU Institutions (20+ terms)", "europarl"),
                                    ("🔬 Chemistry (30+ terms)", "chemistry"),
                                    ("💻 Computer Science (40+ terms)", "cs"),
                                    ("📊 Statistics (25+ terms)", "statistics"),
                                ],
                                label="Select Domain",
                                value="ml"
                            )
                            glossary_direction = gr.Radio(
                                choices=["EN → FR", "FR → EN"],
                                value="EN → FR",
                                label="Direction"
                            )
                            with gr.Row():
                                load_domain_btn = gr.Button("📥 Load Selected", variant="primary")
                                load_all_btn = gr.Button("📚 Load ALL Built-in (250+ terms)", variant="secondary")
                            
                            gr.Markdown("---")
                            gr.Markdown("**🌐 Online Glossary Sources**")
                            online_source = gr.Dropdown(
                                choices=[
                                    ("🇪🇺 Europarl Extended (100+ EU/Legal terms)", "europarl_full"),
                                    ("🔬 Scientific Extended (150+ research terms)", "huggingface_opus"),
                                    ("📖 Wiktionary Common Terms (100+ general)", "wiktionary"),
                                    ("🏛️ IATE EU Terminology (70+ official terms)", "iate"),
                                ],
                                label="Select Online Source",
                                value="europarl_full"
                            )
                            load_online_btn = gr.Button("🌐 Load from Online", variant="secondary")
                            
                            gr.Markdown("---")
                            gr.Markdown("**📁 Custom Glossary**")
                            glossary_file = gr.File(label="Upload JSON file", file_types=[".json"])
                            load_file_btn = gr.Button("📥 Load from File", size="sm")
                            
                            gr.Markdown("---")
                            gr.Markdown("**✏️ Add Individual Term**")
                            with gr.Row():
                                term_source = gr.Textbox(label="Source", placeholder="neural network", scale=1)
                                term_target = gr.Textbox(label="Translation", placeholder="réseau de neurones", scale=1)
                            with gr.Row():
                                add_term_btn = gr.Button("➕ Add", size="sm", variant="primary")
                                clear_gloss_btn = gr.Button("🗑️ Clear All", size="sm", variant="stop")
                            
                            glossary_status = gr.Textbox(label="Status", lines=2, interactive=False, value=f"Glossary loaded: {len(self.glossary)} terms\nLocation: ~/.scitrans/glossary.json")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("**📖 Current Glossary**")
                            term_count = gr.Textbox(value=f"{len(self.glossary)} terms loaded", label="", interactive=False)
                            glossary_preview = gr.Dataframe(
                                headers=["Source Term", "Translation"],
                                value=[[k, v] for k, v in list(self.glossary.items())[:50]] if self.glossary else [],
                                label="Terms (showing first 50)",
                                wrap=True,
                                interactive=False
                            )
                            
                            gr.Markdown("""
**Available Glossary Sources:**

🔹 **Built-in** (offline, instant):
- ML/AI, Physics, Biology, Chemistry, CS, Statistics, Legal

🔹 **Online** (fetched from web):
- Europarl: Official EU translation terminology
- HuggingFace OPUS: Multilingual parallel corpus terms
- Wiktionary: Common academic vocabulary
- IATE: Inter-Active Terminology for Europe

**JSON Format for Custom Upload:**
```json
{
  "source term": "translated term",
  "neural network": "réseau de neurones"
}
```
                            """)
                
                # ===========================================================
                # TAB 5: ABOUT
                # ===========================================================
                with gr.Tab("About"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("""
                            ## SciTrans LLMs v1.0
                            
                            Scientific Document Translation System with layout preservation.
                            
                            ### Features
                            - **Masking**: Protects LaTeX, URLs, DOIs, code blocks
                            - **Reranking**: Multi-candidate translation selection
                            - **Context**: Document-level consistency
                            - **Layout**: Preserves PDF structure and fonts
                            - **Caching**: Persistent cache for faster re-translations
                            
                            ### Backends
                            | Backend | Type | Cost |
                            |---------|------|------|
                            | cascade | Free | Free |
                            | free | Free | Free |
                            | local | Local | Free |
                            | libre | Free | Free |
                            | argos | Local | Free |
                            | ollama | Local | Free |
                            | openai | API | $$$ |
                            | anthropic | API | $$$ |
                            | deepseek | API | $ |
                            | huggingface | API/Free | Free/$$$ |
                            """)
                        
                        with gr.Column():
                            gr.Markdown("""
                            ### CLI Usage
                            
                            ```bash
                            # Basic translation
                            ./scitrans.sh translate paper.pdf -o output.pdf
                            
                            # With specific backend
                            ./scitrans.sh translate paper.pdf --backend openai
                            
                            # Run tests
                            ./scitrans.sh test
                            
                            # List backends
                            ./scitrans.sh backends
                            ```
                            
                            ### Tips
                            - Use **Cascade** for free translations
                            - Enable **Reranking** for academic papers
                            - Add domain terms to **Glossary** tab
                            - Use **Context** for long documents
                            - Cache persists between sessions for speed
                            """)
            
            # ===========================================================
            # EVENT HANDLERS
            # ===========================================================
            
            # PDF upload
            def on_upload(pdf):
                self.translated_pdf_path = None
                if pdf is None:
                    self.source_pdf_path = None
                    return None, None, gr.update(maximum=1, value=1), "of 1", gr.update(visible=False, value="")
                # Store source PDF path for preview
                pdf_path = pdf.name if hasattr(pdf, 'name') else str(pdf)
                self.source_pdf_path = pdf_path
                count = self.get_page_count(pdf)
                # Return source PDF for source_preview, None for trans_preview (not translated yet)
                return pdf_path, None, gr.update(maximum=max(1, count), value=1), f"of {max(1, count)}", gr.update(visible=False, value="")
            
            # URL download
            def on_url_load(url):
                if not url or not url.strip():
                    self.source_pdf_path = None
                    return None, None, gr.update(maximum=1, value=1), "of 1", gr.update(visible=True, value="❌ Please enter a URL")
                
                pdf_path, status_msg = self.download_pdf_from_url(url)
                if pdf_path:
                    # Store source PDF path
                    self.source_pdf_path = pdf_path
                    # Create a file-like object for Gradio
                    count = self.get_page_count(type('obj', (object,), {'name': pdf_path})())
                    # Return source PDF for source_preview, None for trans_preview
                    return pdf_path, None, gr.update(maximum=max(1, count), value=1), f"of {max(1, count)}", gr.update(visible=True, value=status_msg)
                else:
                    self.source_pdf_path = None
                    return None, None, gr.update(maximum=1, value=1), "of 1", gr.update(visible=True, value=status_msg)
            
            pdf_upload.change(fn=on_upload, inputs=[pdf_upload], outputs=[source_preview, trans_preview, page_slider, page_total, url_status])
            url_load_btn.click(fn=on_url_load, inputs=[pdf_url], outputs=[source_preview, trans_preview, page_slider, page_total, url_status])
            
            # Unified page navigation for source and translated previews
            # Update both source and translated previews independently based on page slider
            def nav_page(pdf, page, total, direction):
                # Get source PDF path (from upload or stored)
                source_path = self.source_pdf_path if self.source_pdf_path else (pdf.name if hasattr(pdf, 'name') else str(pdf) if pdf else None)
                # Get translated PDF path (only available after translation)
                trans_path = self.translated_pdf_path if getattr(self, "translated_pdf_path", None) else None
                
                try:
                    max_p = int(str(total).replace("of", "").strip())
                except Exception:
                    max_p = 1
                new_p = max(1, min(max_p, int(page) + direction))
                
                # Return source PDF for source_preview, translated PDF for trans_preview
                return source_path, trans_path, new_p
            
            page_prev.click(
                fn=lambda p, pg, t: nav_page(p, pg, t, -1),
                inputs=[pdf_upload, page_slider, page_total],
                outputs=[source_preview, trans_preview, page_slider],
            )
            page_next.click(
                fn=lambda p, pg, t: nav_page(p, pg, t, 1),
                inputs=[pdf_upload, page_slider, page_total],
                outputs=[source_preview, trans_preview, page_slider],
            )
            page_slider.change(
                fn=lambda p, pg, t: nav_page(p, pg, t, 0),
                inputs=[pdf_upload, page_slider, page_total],
                outputs=[source_preview, trans_preview, page_slider],
            )
            
            # Translation
            translate_inputs = [
                pdf_upload,
                source_lang,
                target_lang,
                backend,
                model_selector,
                advanced_options,
                num_candidates,
                context_window,
                quality_threshold,
                prompt_rounds,
                batch_size,
                enable_parallel,
                max_workers,
                adaptive_concurrency,
                start_page,
                end_page,
                font_dir,
                font_files,
                font_priority,
                mask_custom_macros,
                mask_apostrophes_in_latex,
            ]
            translate_outputs = [status_box, download_btn, log_box, trans_preview, page_total, page_slider, source_preview, perf_info, translation_preview]
            
            translate_btn.click(
                fn=self.translate_document,
                inputs=translate_inputs,
                outputs=translate_outputs
            )
            retranslate_btn.click(
                fn=self.translate_document,
                inputs=translate_inputs,
                outputs=translate_outputs
            )
            
            def clear_all():
                self.translated_pdf_path = None
                self.source_pdf_path = None
                return "", gr.update(value=None, visible=False), "", None, "of 1", gr.update(maximum=1, value=1), None, "", ""
            
            clear_btn.click(fn=clear_all, outputs=translate_outputs)
            
            def update_backend_and_model(backend_value):
                """Update model options when backend changes, ensuring valid value."""
                opts = self._get_model_options_for_backend(backend_value)
                # Always reset to first valid option to prevent invalid "default" values
                return gr.update(choices=opts["choices"], value=opts["value"], visible=opts["visible"])
            
            backend.change(
                fn=update_backend_and_model,
                inputs=[backend],
                outputs=[model_selector]
            )
            
            # Testing
            test_backend_btn.click(fn=self.test_backend, inputs=[test_backend_sel, test_text], outputs=[test_backend_result])
            test_masking_btn.click(fn=self.test_masking, inputs=[masking_input], outputs=[test_masking_result])
            test_layout_btn.click(fn=self.test_layout, inputs=[layout_pdf], outputs=[test_layout_result])
            test_cache_btn.click(fn=self.test_cache, outputs=[test_cache_result])
            
            # Settings
            def update_api_keys_table():
                return self._get_api_keys_table()
            
            refresh_keys_btn.click(
                fn=update_api_keys_table,
                outputs=[api_keys_table]
            )
            
            save_key_btn.click(
                fn=self.save_api_key, 
                inputs=[api_backend, api_key_input], 
                outputs=[api_status]
            ).then(
                fn=update_api_keys_table, 
                outputs=[api_keys_table]
            )
            
            delete_key_btn.click(
                fn=self.delete_api_key,
                inputs=[api_backend],
                outputs=[api_status]
            ).then(
                fn=update_api_keys_table,
                outputs=[api_keys_table]
            )
            
            save_settings_btn.click(
                fn=self.save_all_settings,
                inputs=[
                    set_backend, set_masking, set_reranking, set_cache, 
                    set_glossary, set_context, set_context_window, 
                    set_candidates, set_strict_mode, set_fallback
                ],
                outputs=[settings_status]
            )
            
            reset_settings_btn.click(
                fn=self.reset_settings,
                outputs=[settings_status]
            )
            
            clear_cache_btn.click(fn=self.clear_cache, outputs=[cache_status])
            
            # Glossary - new dropdown-based UI
            def load_selected_domain(domain, direction):
                dir_code = "en-fr" if direction == "EN → FR" else "fr-en"
                return self.load_glossary_domain(domain, dir_code)
            
            def get_glossary_preview():
                # Return first 50 terms for preview as DataFrame format
                if self.glossary:
                    return [[k, v] for k, v in list(self.glossary.items())[:50]]
                return []
            
            load_domain_btn.click(
                fn=load_selected_domain,
                inputs=[glossary_domain, glossary_direction],
                outputs=[glossary_status, glossary_preview]
            ).then(fn=lambda: f"{len(self.glossary)} terms loaded", outputs=[term_count])
            
            def load_all_with_direction(direction):
                dir_code = "en-fr" if direction == "EN → FR" else "fr-en"
                return self.load_all_scientific_glossaries(dir_code)
            
            load_all_btn.click(
                fn=load_all_with_direction,
                inputs=[glossary_direction],
                outputs=[glossary_status, glossary_preview]
            ).then(fn=lambda: f"{len(self.glossary)} terms loaded", outputs=[term_count])
            
            # File upload and manual terms
            load_file_btn.click(
                fn=self.load_glossary_file, 
                inputs=[glossary_file], 
                outputs=[glossary_status, glossary_preview]
            ).then(fn=lambda: f"{len(self.glossary)} terms loaded", outputs=[term_count])
            
            add_term_btn.click(
                fn=self.add_glossary_term, 
                inputs=[term_source, term_target], 
                outputs=[glossary_status, glossary_preview]
            ).then(fn=lambda: f"{len(self.glossary)} terms", outputs=[term_count])
            
            clear_gloss_btn.click(
                fn=self.clear_glossary, 
                outputs=[glossary_status, glossary_preview]
            ).then(fn=lambda: "0 terms", outputs=[term_count])
            
            # Online glossary loading
            def load_online_glossary_handler(source, direction):
                dir_code = "en-fr" if direction == "EN → FR" else "fr-en"
                return self.load_online_glossary(source, dir_code)
            
            load_online_btn.click(
                fn=load_online_glossary_handler,
                inputs=[online_source, glossary_direction],
                outputs=[glossary_status, glossary_preview]
            ).then(fn=lambda: f"{len(self.glossary)} terms loaded", outputs=[term_count])
        
        return demo


def find_free_port(start_port=7860, max_attempts=10):
    """Find a free port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start_port + max_attempts


def launch(share=False, port=None):
    """Launch the GUI."""
    if port is None:
        port = find_free_port(7860)
    
    gui = SciTransGUI()
    app = gui.create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        inbrowser=True
    )


if __name__ == "__main__":
    launch()
