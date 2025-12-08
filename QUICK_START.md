# 🚀 Quick Start Guide

## ⚡ **Start Using in 30 Seconds**

### **1. Test It Works**
```bash
scitrans test --backend cascade
```

**Expected Output:**
```
✓ Translation successful!
Result: Bonjour le monde
Backend: cascade
```

### **2. Translate Your First Document**
```bash
scitrans wizard
```

Follow the prompts - it's that easy! ✅

---

## 📖 **Three Ways to Use**

### **Method 1: Interactive Wizard** ⭐ **EASIEST**

```bash
scitrans wizard
```

**What happens:**
1. 📁 Select your PDF file
2. 🔧 Choose backend (cascade is FREE)
3. 🌍 Set languages (default: en → fr)
4. ⚙️ Enable options (masking recommended)
5. ✅ Done! Translated file is saved

**Perfect for:** First-time users, quick translations

### **Method 2: Direct Command**

```bash
scitrans translate paper.pdf --backend cascade -o translated.pdf
```

**Perfect for:** Scripts, automation, batch processing

### **Method 3: Python API**

```python
from scitran.core.pipeline import TranslationPipeline, PipelineConfig

config = PipelineConfig(backend="cascade", enable_masking=True)
pipeline = TranslationPipeline(config)
result = pipeline.translate_pdf("input.pdf", "output.pdf")
```

**Perfect for:** Custom integration, advanced users

---

## 🆓 **FREE Backends**

### **CASCADE** ⭐ **RECOMMENDED**
```bash
scitrans test --backend cascade
```
- **Cost:** FREE
- **Quality:** Good
- **Speed:** Fast
- **Setup:** None
- **Best for:** Everything!

### **FREE (Google Translate)**
```bash
scitrans test --backend free
```
- **Cost:** FREE
- **Quality:** Good
- **Speed:** Very fast
- **Setup:** None
- **Best for:** Quick translations

**Both work immediately with NO API keys!** 🎉

---

## 💰 **Paid Backends** (Optional)

### **DeepSeek** - Most Cost-Effective
```bash
export DEEPSEEK_API_KEY="your-key"
scitrans test --backend deepseek
```
- **Cost:** $0.14 per 1M tokens (~$0.004 per page)
- **Quality:** Excellent
- **Best for:** Bulk experiments

### **OpenAI** - Best Quality
```bash
export OPENAI_API_KEY="sk-your-key"
scitrans test --backend openai
```
- **Cost:** $2.50 per 1M tokens (~$0.10 per page)
- **Quality:** Best available
- **Best for:** Final results

### **Anthropic** - Long Documents
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key"
scitrans test --backend anthropic
```
- **Cost:** $3.00 per 1M tokens (~$0.12 per page)
- **Quality:** Best available
- **Best for:** Long papers

**See `API_KEYS_SETUP.md` for detailed instructions.**

---

## ⚙️ **Common Options**

### **Enable LaTeX Masking** (Recommended for scientific papers)
```bash
scitrans translate paper.pdf --backend cascade --masking
```

Protects:
- Mathematical formulas (`$...$`, `$$...$$`)
- LaTeX commands (`\ref`, `\cite`, etc.)
- Code blocks
- URLs

### **Enable Quality Reranking** (Better quality, slower)
```bash
scitrans translate paper.pdf --backend cascade --reranking --candidates 5
```

Generates 5 translations and selects the best one.

### **Set Languages**
```bash
scitrans translate paper.pdf --backend cascade -s en -t de
```
- `-s` = source language (en, fr, de, es, zh, etc.)
- `-t` = target language

---

## 📊 **Examples**

### **Example 1: Quick Translation (FREE)**
```bash
# Use CASCADE - no setup needed!
scitrans translate research_paper.pdf --backend cascade
```

**Time:** 2-5 minutes for 10-page paper  
**Cost:** $0  
**Quality:** Good for most purposes

### **Example 2: Batch Translate Multiple Files**
```bash
# Translate all PDFs in current directory
for file in *.pdf; do
  scitrans translate "$file" --backend cascade -o "translated_$file"
done
```

### **Example 3: Best Quality Translation**
```bash
# Set API key first
export OPENAI_API_KEY="sk-your-key"

# Translate with GPT-4o + all optimizations
scitrans translate paper.pdf \
  --backend openai \
  --model gpt-4o \
  --masking \
  --reranking \
  --candidates 5 \
  -o final_translation.pdf
```

**Cost:** ~$0.50 for 20-page paper  
**Quality:** Best possible

---

## 🔍 **Check Status**

### **List All Backends**
```bash
scitrans backends
```

Output shows:
- ✅ Available backends (ready to use)
- ❌ Not configured (need API key)

### **Detailed Backend Info**
```bash
scitrans backends --detailed
```

Shows:
- Cost comparison
- Quality ratings
- Speed ratings
- Setup requirements

---

## 🐛 **Troubleshooting**

### **Issue: Command not found**
```bash
# Solution: Reinstall
cd /Users/kv.kn/Desktop/Research/SciTrans-LLMs_NEW
pip install -e . --force-reinstall --no-deps
```

### **Issue: Backend not configured**
```bash
# Check which backends work
scitrans backends

# Use CASCADE (always works)
scitrans test --backend cascade
```

### **Issue: API key not working**
```bash
# Check if key is set
echo $OPENAI_API_KEY

# If empty, set it:
export OPENAI_API_KEY="sk-your-key"

# Test again
scitrans test --backend openai
```

---

## 📚 **Documentation**

| File | Description |
|------|-------------|
| `docs/API_KEYS_SETUP.md` | Complete API key setup guide |
| `docs/TESTING_GUIDE.md` | Testing and benchmarking guide |
| `CODEBASE_ANALYSIS.md` | Technical analysis and improvements |
| `README.md` | Project overview |

---

## 🎯 **Recommended Workflow**

### **For First Time:**
```bash
# 1. Test it works
scitrans test --backend cascade

# 2. Use interactive mode
scitrans wizard

# 3. Done! ✅
```

### **For Regular Use:**
```bash
# Direct translation with CASCADE (FREE)
scitrans translate paper.pdf --backend cascade --masking
```

### **For Production/Thesis:**
```bash
# Set up DeepSeek (cheap + good)
export DEEPSEEK_API_KEY="your-key"

# Translate with quality options
scitrans translate paper.pdf \
  --backend deepseek \
  --masking \
  --reranking \
  -o final.pdf
```

---

## ⏱️ **How Long Does It Take?**

| Document Size | CASCADE (FREE) | DeepSeek | OpenAI |
|---------------|----------------|----------|--------|
| 5 pages | 1-2 min | 1-2 min | 1-2 min |
| 20 pages | 5-8 min | 4-6 min | 3-5 min |
| 100 pages | 20-30 min | 15-25 min | 12-20 min |

**With reranking:** Add 2-3x more time (but better quality)

---

## 💡 **Pro Tips**

### **Tip 1: Start with FREE**
```bash
# Test with CASCADE first
scitrans translate paper.pdf --backend cascade

# If quality is good enough, you're done! $0 cost
```

### **Tip 2: Use Masking for Scientific Papers**
```bash
# Always enable --masking for LaTeX documents
scitrans translate paper.pdf --backend cascade --masking
```

### **Tip 3: Batch Process Smartly**
```bash
# Use CASCADE for initial drafts (FREE)
# Use DeepSeek for experiments (CHEAP)
# Use OpenAI for final version (BEST)

backends="cascade deepseek openai"
for backend in $backends; do
  scitrans translate paper.pdf --backend $backend -o "${backend}.pdf"
done
```

### **Tip 4: Set Default Options**
```bash
# Create alias in ~/.zshrc
alias translate='scitrans translate --backend cascade --masking'

# Then simply use:
translate paper.pdf
```

---

## 🎓 **For Students/Researchers**

### **Budget-Friendly Approach:**

```bash
# Phase 1: Testing - Use CASCADE (FREE)
scitrans translate sample.pdf --backend cascade

# Phase 2: Main work - Use DeepSeek (CHEAP)
export DEEPSEEK_API_KEY="your-key"
scitrans translate thesis.pdf --backend deepseek --masking

# Phase 3: Final polish - Use OpenAI (BEST)
export OPENAI_API_KEY="sk-your-key"
scitrans translate final.pdf --backend openai --masking --reranking

# Total cost: ~$5-10 for entire thesis!
```

---

## ✅ **Summary**

### **What Works:**
- ✅ All CLI commands (8/8)
- ✅ CASCADE backend (FREE, recommended)
- ✅ FREE backend (Google Translate)
- ✅ All paid backends (with API keys)
- ✅ Masking, reranking, all features

### **What to Use:**
```bash
# Use this for everything:
scitrans wizard

# Or for quick translations:
scitrans translate paper.pdf --backend cascade
```

### **What to Read:**
1. This file (`QUICK_START.md`) - You're done reading it! ✅
2. `docs/API_KEYS_SETUP.md` - If you want paid backends
3. `README.md` - For project overview and advanced features

---

## 🚀 **Ready?**

```bash
# Start now:
scitrans wizard

# Or quick test:
scitrans test --backend cascade

# Everything works! 🎉
```

**Have fun translating!** 📚✨
