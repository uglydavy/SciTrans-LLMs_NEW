import pytest

from scitran.masking.engine import MaskingEngine, MaskingConfig


def apply_masks(text: str, config: MaskingConfig):
    engine = MaskingEngine(config=config)
    masked = engine.apply_text(text)
    return masked, engine


def test_apostrophes_inside_math():
    text = "Math: $f'(x)$ and \\(g'(y)\\)"
    masked, eng = apply_masks(text, MaskingConfig())
    # placeholders should replace math segments
    assert "<<" in masked
    assert "latex_apostrophe" in eng.mask_counter


def test_text_ops_and_newcommand():
    text = "\\newcommand{\\foo}{\\text{bar}} $\\mathrm{X} + \\mathbb{R}$"
    masked, eng = apply_masks(text, MaskingConfig(mask_custom_macros=True))
    assert any(k.startswith("latex_newcommand") for k in eng.mask_counter)
    assert "latex_text_ops" in eng.mask_counter or "latex_inline" in eng.mask_counter


def test_declare_math_operator():
    text = "\\DeclareMathOperator{\\Var}{Var} $\\Var(X)$"
    masked, eng = apply_masks(text, MaskingConfig(mask_custom_macros=True))
    assert "latex_declare_operator" in eng.mask_counter


def test_provide_command_optional():
    text = "\\providecommand{\\foo}{\\text{bar}} $\\foo$"
    masked, eng = apply_masks(text, MaskingConfig(mask_custom_macros=True))
    assert "latex_provide_command" in eng.mask_counter or "latex_inline" in eng.mask_counter

