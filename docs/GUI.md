# GUI Guide

Launch the Gradio interface and translate PDFs with layout preservation.

## Launch
```bash
./scitrans gui
# Opens in browser; defaults to port 7860 or next free port.
```

## Key options exposed in GUI
- Backend selection (cascade/openai/anthropic/deepseek/free/huggingface/ollama)
- Strict mode (default on): abort if any block missing translation; downloadable JSON report
- Max retries and fallback backend
- Glossary selection: bundled domains + custom JSON upload
- Masking toggle (for ablations only; keep on for real use)
- Refinement and context toggles

## Tips
- Use strict mode for thesis runs to avoid partial PDFs.
- Provide a glossary for domain terms; bundled domains are available.
- If translation seems incomplete, increase retries or enable fallback backend.

