# -*- coding: utf-8 -*-
"""
Artifact generation for transparency and debugging.

Generates machine-readable artifacts for each translation run:
- extraction.json: Blocks extracted, page, bbox, type, text
- masking.json: Token counts, validation
- translation.json: Per-block status + failure reason
- run.log: Structured log
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from scitran.core.models import Document, Block, BlockType
from scitran.core.validator import ValidationResult

logger = logging.getLogger(__name__)


class ArtifactGenerator:
    """Generate artifacts for translation runs."""
    
    def __init__(self, run_id: str, output_dir: Optional[Path] = None):
        self.run_id = run_id
        self.output_dir = output_dir or Path(f"artifacts/{run_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        self.log_file = self.output_dir / "run.log"
        self.log_handle = open(self.log_file, 'w', encoding='utf-8')
        
        self.log(f"Artifact generation started for run {run_id}")
    
    def log(self, message: str):
        """Write to structured log."""
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] {message}\n"
        self.log_handle.write(log_line)
        self.log_handle.flush()
    
    def save_extraction(self, document: Document):
        """Save extraction artifacts (blocks, page, bbox, type, text)."""
        extraction_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "document_id": document.document_id,
            "source_path": document.source_path,
            "total_blocks": len(document.all_blocks),
            "translatable_blocks": len(document.translatable_blocks),
            "blocks": []
        }
        
        for block in document.all_blocks:
            block_data = {
                "block_id": block.block_id,
                "block_type": block.block_type.name if hasattr(block.block_type, 'name') else str(block.block_type),
                "is_translatable": block.is_translatable,
                "source_text": block.source_text,
                "text_length": len(block.source_text),
                "bbox": {
                    "page": block.bbox.page,
                    "x0": block.bbox.x0,
                    "y0": block.bbox.y0,
                    "x1": block.bbox.x1,
                    "y1": block.bbox.y1,
                } if block.bbox else None,
                "font": {
                    "family": block.font.family,
                    "size": block.font.size,
                    "weight": block.font.weight,
                    "style": block.font.style,
                    "color": block.font.color,
                    "alignment": block.font.alignment if hasattr(block.font, 'alignment') else None,
                } if block.font else None,
            }
            extraction_data["blocks"].append(block_data)
        
        # Save to JSON
        extraction_file = self.output_dir / "extraction.json"
        with open(extraction_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"Extraction artifacts saved: {extraction_file}")
        logger.info(f"Extraction artifacts: {extraction_file}")
    
    def save_masking(self, document: Document, masking_engine: Optional[Any] = None):
        """Save masking artifacts (token counts, validation)."""
        masking_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "total_masks": sum(len(block.masks) for block in document.all_blocks),
            "blocks_with_masks": sum(1 for block in document.all_blocks if block.masks),
            "masks_by_type": {},
            "blocks": []
        }
        
        # Count masks by type
        for block in document.all_blocks:
            for mask in block.masks:
                mask_type = mask.mask_type
                masking_data["masks_by_type"][mask_type] = masking_data["masks_by_type"].get(mask_type, 0) + 1
        
        # Per-block masking info
        for block in document.all_blocks:
            if not block.masks:
                continue
            
            block_data = {
                "block_id": block.block_id,
                "num_masks": len(block.masks),
                "masked_text": block.masked_text,
                "masks": [
                    {
                        "placeholder": mask.placeholder,
                        "mask_type": mask.mask_type,
                        "original": mask.original,
                        "preserve_formatting": mask.preserve_formatting
                    }
                    for mask in block.masks
                ]
            }
            masking_data["blocks"].append(block_data)
        
        # Add masking engine stats if available
        if masking_engine:
            try:
                masking_data["engine_stats"] = masking_engine.get_statistics()
                masking_data["validation_errors"] = masking_engine.validation_errors
            except:
                pass
        
        # Save to JSON
        masking_file = self.output_dir / "masking.json"
        with open(masking_file, 'w', encoding='utf-8') as f:
            json.dump(masking_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"Masking artifacts saved: {masking_file}")
        logger.info(f"Masking artifacts: {masking_file}")
    
    def save_translation(self, document: Document, validation_result: Optional[ValidationResult] = None):
        """Save translation artifacts (per-block status + failure reason)."""
        translation_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "total_blocks": len(document.all_blocks),
            "translatable_blocks": len(document.translatable_blocks),
            "translated_blocks": sum(1 for b in document.translatable_blocks if b.translated_text),
            "blocks": []
        }
        
        # Add validation summary if available
        if validation_result:
            translation_data["validation"] = {
                "is_valid": validation_result.is_valid,
                "coverage": validation_result.coverage,
                "errors_count": len(validation_result.errors),
                "warnings_count": len(validation_result.warnings),
                "failed_blocks": validation_result.failed_blocks,
                "identity_blocks": validation_result.identity_blocks,
                "missing_masks_count": len(validation_result.missing_masks),
            }
        
        # Per-block translation info
        for block in document.all_blocks:
            block_data = {
                "block_id": block.block_id,
                "block_type": block.block_type.name if hasattr(block.block_type, 'name') else str(block.block_type),
                "is_translatable": block.is_translatable,
                "has_translation": bool(block.translated_text and block.translated_text.strip()),
                "source_length": len(block.source_text),
                "translation_length": len(block.translated_text) if block.translated_text else 0,
            }
            
            # Add metadata if available
            if block.metadata:
                block_data["metadata"] = {
                    "backend": getattr(block.metadata, 'backend', None) or getattr(block.metadata, 'backend_used', None),
                    "status": getattr(block.metadata, 'status', 'unknown'),
                    "failure_reason": getattr(block.metadata, 'failure_reason', None),
                    "finish_reason": getattr(block.metadata, 'finish_reason', None),
                    "was_truncated": getattr(block.metadata, 'was_truncated', False),
                    "retry_count": getattr(block.metadata, 'retry_count', 0),
                }
            
            translation_data["blocks"].append(block_data)
        
        # Save to JSON
        translation_file = self.output_dir / "translation.json"
        with open(translation_file, 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"Translation artifacts saved: {translation_file}")
        logger.info(f"Translation artifacts: {translation_file}")
    
    def close(self):
        """Close log file."""
        if self.log_handle:
            self.log_handle.close()
    
    def __del__(self):
        """Cleanup."""
        try:
            if hasattr(self, 'log_handle') and self.log_handle:
                self.log_handle.close()
        except:
            pass

