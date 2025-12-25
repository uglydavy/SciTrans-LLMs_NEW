# Fixes Applied for Translation and Preview Issues

## Issues Fixed

### 1. **Fallback Translation Using Wrong Translator**
**Problem**: `_fallback_translation()` was using `self.translator` (the primary translator that already failed) instead of creating a separate fallback translator.

**Fix**: Modified `_fallback_translation()` in `scitran/core/pipeline.py` to:
- Check for existing `_fallback_translator` attribute
- Create a new translator for the fallback backend if needed
- Use fallback translator, or primary translator as last resort
- Better error handling with exception wrapping

### 2. **Source Preview Not Showing on Error**
**Problem**: When translation failed, the GUI error handler returned `None` for source preview, so users couldn't see the uploaded PDF.

**Fix**: Modified error handler in `gui/app.py` to:
- Always preserve `source_pdf_path` even on error
- Return source PDF path in error return statement
- Ensure source preview shows even when translation fails

### 3. **Missing API Key Error Messages**
**Problem**: When DeepSeek (or other paid backends) was selected without an API key, the error was unclear.

**Fix**: Added early API key validation in `translate_document()`:
- Check if backend requires API key before starting translation
- Return clear error message with instructions
- Suggest using "free" or "cascade" backends for testing
- Still show source PDF preview even when API key is missing

## Files Modified

1. `scitran/core/pipeline.py`
   - `_fallback_translation()` method: Now uses separate fallback translator

2. `gui/app.py`
   - Error handler: Always returns source PDF path
   - `translate_document()`: Early API key validation with helpful messages

## Testing Recommendations

1. **Test without API key**:
   - Select DeepSeek backend without setting API key
   - Should see clear error message
   - Source PDF preview should still show

2. **Test with free backend**:
   - Select "free" backend
   - Should work without API key
   - Should translate successfully

3. **Test fallback**:
   - Use a backend that fails (e.g., OpenAI without key)
   - Enable fallback backend
   - Should attempt fallback translation

4. **Test previews**:
   - Upload PDF
   - Source preview should show immediately
   - After translation (or error), both previews should be accessible

## Next Steps

If translation still fails:
1. Check API keys are set correctly in Settings tab
2. Check environment variables: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, etc.
3. Try "free" backend first to verify the system works
4. Check logs for detailed error messages



