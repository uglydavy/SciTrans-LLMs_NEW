"""
Output cleaning utilities for LLM translations.

STEP 3: Clean up common LLM output artifacts:
- Remove <think>...</think> wrappers (chain-of-thought)
- Remove "Translation:" prefixes
- Remove code fence wrappers
- Remove leading/trailing quotes
"""

import re
from typing import List


def clean_translation_output(text: str) -> str:
    """
    Clean LLM translation output to extract only the translated text.
    
    Args:
        text: Raw LLM output
        
    Returns:
        Cleaned translation text
    """
    if not text:
        return text
    
    original = text
    
    # 1. Remove <think>...</think> blocks (chain-of-thought reasoning)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Remove <thinking>...</thinking> blocks
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 3. Remove markdown code fences
    # Match ```language\ntext\n``` or ```\ntext\n```
    code_fence_pattern = r'^```(?:\w+)?\s*\n(.*?)\n```\s*$'
    match = re.match(code_fence_pattern, text.strip(), re.DOTALL)
    if match:
        text = match.group(1)
    
    # 4. Remove "Translation:" prefix (case-insensitive)
    text = re.sub(r'^Translation:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Translated text:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Output:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Result:\s*', '', text, flags=re.IGNORECASE)
    
    # 5. Remove leading/trailing quotes if they wrap the entire text
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        # Only remove if they're balanced and wrap the whole thing
        text = text[1:-1]
    
    # 6. Remove any remaining leading/trailing whitespace
    text = text.strip()
    
    # If we removed everything, return original
    if not text:
        return original
    
    return text


def clean_batch_outputs(translations: List[str]) -> List[str]:
    """
    Clean a batch of translation outputs.
    
    Args:
        translations: List of raw LLM outputs
        
    Returns:
        List of cleaned translations
    """
    return [clean_translation_output(t) for t in translations]


def has_reasoning_wrapper(text: str) -> bool:
    """
    Check if text contains reasoning wrappers that should be cleaned.
    
    Args:
        text: Text to check
        
    Returns:
        True if reasoning wrappers detected
    """
    if not text:
        return False
    
    patterns = [
        r'<think>',
        r'<thinking>',
        r'^Translation:',
        r'^```',
    ]
    
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

