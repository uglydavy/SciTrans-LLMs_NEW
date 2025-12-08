# 🔑 API Keys Setup Guide

Complete guide for setting up API keys for all translation backends.

---

## 🆓 **FREE Backends (No API Key Needed)**

These work immediately without any setup:

### **1. CASCADE** ⭐ RECOMMENDED
```bash
scitrans test --backend cascade
# No API key needed! ✅
```

### **2. FREE (Google Translate)**
```bash
scitrans test --backend free
# No API key needed! ✅
```

### **3. OLLAMA (Local)**
```bash
# Install Ollama first
brew install ollama

# Start service
ollama serve

# Pull model
ollama pull llama3.1

# Then use
scitrans test --backend ollama
# No API key needed! ✅
```

---

## 💰 **PAID Backends (API Key Required)**

### **1. OpenAI (GPT-4, GPT-4o)** 

#### **Get API Key:**
1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

#### **Set API Key:**

**Option A: Environment Variable (Recommended)**
```bash
# Add to ~/.zshrc or ~/.bashrc
export OPENAI_API_KEY="sk-your-key-here"

# Reload
source ~/.zshrc

# Verify
echo $OPENAI_API_KEY
```

**Option B: .env File**
```bash
# Create .env file in project root
cd /Users/kv.kn/Desktop/Research/SciTrans-LLMs_NEW
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

**Option C: Direct in Code**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
```

#### **Test It:**
```bash
scitrans test --backend openai --sample "Hello world"

# Expected output:
# ✓ Translation successful!
# Result: Bonjour le monde
# Backend: openai
# Model: gpt-4o
```

#### **Pricing:**
- GPT-4o: $2.50 per 1M input tokens, $10 per 1M output tokens
- GPT-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens
- GPT-3.5-turbo: $0.50 per 1M input tokens, $1.50 per 1M output tokens

---

### **2. Anthropic (Claude)** 

#### **Get API Key:**
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Go to "API Keys" section
4. Click "Create Key"
5. Copy the key (starts with `sk-ant-...`)

#### **Set API Key:**

**Option A: Environment Variable (Recommended)**
```bash
# Add to ~/.zshrc or ~/.bashrc
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Reload
source ~/.zshrc

# Verify
echo $ANTHROPIC_API_KEY
```

**Option B: .env File**
```bash
# Add to .env file
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> .env
```

#### **Test It:**
```bash
scitrans test --backend anthropic --sample "Hello world"

# Expected output:
# ✓ Translation successful!
# Result: Bonjour le monde
# Backend: anthropic
# Model: claude-3-5-sonnet-20241022
```

#### **Pricing:**
- Claude 3.5 Sonnet: $3 per 1M input tokens, $15 per 1M output tokens
- Claude 3 Opus: $15 per 1M input tokens, $75 per 1M output tokens
- Claude 3 Haiku: $0.25 per 1M input tokens, $1.25 per 1M output tokens

---

### **3. DeepSeek (Cost-Effective)** 

#### **Get API Key:**
1. Go to https://platform.deepseek.com/
2. Sign up or log in
3. Go to "API Keys"
4. Create new API key
5. Copy the key

#### **Set API Key:**

**Option A: Environment Variable (Recommended)**
```bash
# Add to ~/.zshrc or ~/.bashrc
export DEEPSEEK_API_KEY="your-key-here"

# Reload
source ~/.zshrc

# Verify
echo $DEEPSEEK_API_KEY
```

**Option B: .env File**
```bash
# Add to .env file
echo "DEEPSEEK_API_KEY=your-key-here" >> .env
```

#### **Test It:**
```bash
scitrans test --backend deepseek --sample "Hello world"

# Expected output:
# ✓ Translation successful!
# Result: Bonjour le monde
# Backend: deepseek
# Model: deepseek-chat
```

#### **Pricing:**
- DeepSeek Chat: $0.14 per 1M input tokens, $0.28 per 1M output tokens
- **Most cost-effective option!** 💰

---

### **4. HuggingFace (Optional - For Higher Rate Limits)** 

**Note:** HuggingFace has a free tier without API key, but rate-limited.

#### **Get API Key (Optional):**
1. Go to https://huggingface.co/settings/tokens
2. Sign up or log in
3. Click "New token"
4. Select "Read" access
5. Copy the token (starts with `hf_...`)

#### **Set API Key:**

**Option A: Environment Variable**
```bash
# Add to ~/.zshrc or ~/.bashrc
export HUGGINGFACE_API_KEY="hf_your-key-here"

# Reload
source ~/.zshrc
```

**Option B: .env File**
```bash
# Add to .env file
echo "HUGGINGFACE_API_KEY=hf_your-key-here" >> .env
```

#### **Pricing:**
- **FREE tier:** 1000 requests/month (no credit card)
- **Pro tier:** $9/month for higher limits
- **Enterprise:** Custom pricing

---

## 📝 **Complete Setup Example**

### **Setup All Keys at Once:**

```bash
# Edit your shell config
nano ~/.zshrc  # or ~/.bashrc

# Add these lines:
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export HUGGINGFACE_API_KEY="hf_your-huggingface-key"

# Save and reload
source ~/.zshrc

# Verify all keys
echo "OpenAI: $OPENAI_API_KEY"
echo "Anthropic: $ANTHROPIC_API_KEY"
echo "DeepSeek: $DEEPSEEK_API_KEY"
echo "HuggingFace: $HUGGINGFACE_API_KEY"
```

### **Or Use .env File:**

```bash
# Create .env file
cd /Users/kv.kn/Desktop/Research/SciTrans-LLMs_NEW

cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key
HUGGINGFACE_API_KEY=hf_your-huggingface-key
EOF

# Load .env in Python
pip install python-dotenv

# In your code:
from dotenv import load_dotenv
load_dotenv()
```

---

## ✅ **Verify Setup**

Test all backends to see which ones are configured:

```bash
# Check all backends
scitrans backends

# Output shows:
# ✓ Available cascade (multi-service)          # FREE
# ✓ Available free (google)                    # FREE
# ✓ Available huggingface (...)                # FREE
# ✓ Available ollama (llama3.1)                # FREE
# ✗ Not configured deepseek                    # Needs key
# ✗ Not configured openai                      # Needs key
# ✗ Not configured anthropic                   # Needs key
```

### **Test Each Backend:**

```bash
# Test FREE backends (work without keys)
scitrans test --backend cascade
scitrans test --backend free

# Test PAID backends (need keys)
scitrans test --backend openai      # Requires OPENAI_API_KEY
scitrans test --backend anthropic   # Requires ANTHROPIC_API_KEY
scitrans test --backend deepseek    # Requires DEEPSEEK_API_KEY
```

---

## 🔒 **Security Best Practices**

### **1. Never Commit API Keys**

```bash
# Add .env to .gitignore
echo ".env" >> .gitignore
echo "*.env" >> .gitignore

# Check what would be committed
git status

# Make sure .env is ignored
```

### **2. Use Environment Variables**

```bash
# ✅ GOOD - Environment variable
export OPENAI_API_KEY="sk-..."

# ❌ BAD - Hardcoded in code
api_key = "sk-..."  # DON'T DO THIS!
```

### **3. Rotate Keys Regularly**

- Create new keys every 90 days
- Delete old keys immediately
- Use separate keys for dev/prod

### **4. Limit Key Permissions**

- OpenAI: Set usage limits in dashboard
- Anthropic: Use workspaces for team access
- DeepSeek: Enable IP restrictions if available

---

## 💡 **Usage Examples**

### **Example 1: Use OpenAI for Best Quality**

```bash
# Set key
export OPENAI_API_KEY="sk-your-key"

# Translate with GPT-4o
scitrans translate paper.pdf \
  --backend openai \
  --model gpt-4o \
  --masking \
  --candidates 5 \
  --reranking \
  -o translated.pdf

# Cost: ~$0.10 for 20-page paper
```

### **Example 2: Use DeepSeek for Cost-Effective**

```bash
# Set key
export DEEPSEEK_API_KEY="your-key"

# Translate with DeepSeek
scitrans translate paper.pdf \
  --backend deepseek \
  --masking \
  -o translated.pdf

# Cost: ~$0.01 for 20-page paper (20x cheaper!)
```

### **Example 3: Use CASCADE for FREE**

```bash
# No key needed!
scitrans translate paper.pdf \
  --backend cascade \
  --masking \
  -o translated.pdf

# Cost: $0 (completely free!)
```

### **Example 4: Batch Processing with Different Backends**

```bash
# Set all keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="..."

# Translate with multiple backends
for backend in cascade deepseek openai anthropic; do
  echo "Translating with $backend..."
  scitrans translate paper.pdf \
    --backend $backend \
    --masking \
    -o output_${backend}.pdf
done

# Compare results!
```

---

## 📊 **Cost Comparison**

For a typical 20-page scientific paper (~10,000 tokens):

| Backend | Input Cost | Output Cost | Total | Quality |
|---------|-----------|-------------|-------|---------|
| **CASCADE** | $0 | $0 | **$0** | ⭐⭐⭐ |
| **FREE** | $0 | $0 | **$0** | ⭐⭐⭐ |
| **DeepSeek** | $0.001 | $0.003 | **$0.004** | ⭐⭐⭐⭐ |
| **GPT-4o-mini** | $0.002 | $0.006 | **$0.008** | ⭐⭐⭐⭐ |
| **GPT-4o** | $0.025 | $0.100 | **$0.125** | ⭐⭐⭐⭐⭐ |
| **Claude 3.5** | $0.030 | $0.150 | **$0.180** | ⭐⭐⭐⭐⭐ |

**Recommendation for thesis work:**
- Start with **CASCADE** (free) for testing
- Use **DeepSeek** for main experiments (cheap + good)
- Use **GPT-4o** for final version (best quality)

---

## 🐛 **Troubleshooting**

### **Issue 1: "Not configured" Error**

```bash
# Check if key is set
echo $OPENAI_API_KEY

# If empty, key not set
# Solution: Set the key
export OPENAI_API_KEY="sk-your-key"
```

### **Issue 2: "Invalid API key" Error**

```bash
# Check key format
echo $OPENAI_API_KEY

# OpenAI keys start with: sk-
# Anthropic keys start with: sk-ant-
# HuggingFace keys start with: hf_

# If wrong, get new key from dashboard
```

### **Issue 3: "Rate limit exceeded" Error**

```bash
# Solution 1: Wait a few minutes

# Solution 2: Use different backend
scitrans test --backend cascade  # Free alternative

# Solution 3: Upgrade plan
# OpenAI: Increase usage limits
# Anthropic: Add payment method
```

### **Issue 4: Keys Work in Terminal but Not in Scripts**

```bash
# Problem: Keys not loaded in script environment

# Solution 1: Source shell config in script
#!/bin/bash
source ~/.zshrc
scitrans translate paper.pdf --backend openai

# Solution 2: Use .env file with python-dotenv
from dotenv import load_dotenv
load_dotenv()

# Solution 3: Export keys before running
export OPENAI_API_KEY="sk-..."
python my_script.py
```

---

## 📚 **Quick Reference**

### **Check Backend Status:**
```bash
scitrans backends
```

### **Test Backend:**
```bash
scitrans test --backend [BACKEND_NAME]
```

### **Set Key:**
```bash
export [BACKEND]_API_KEY="your-key"
```

### **Verify Key:**
```bash
echo $[BACKEND]_API_KEY
```

### **Backend Names:**
- `cascade` - FREE
- `free` - FREE
- `ollama` - FREE (local)
- `deepseek` - Needs DEEPSEEK_API_KEY
- `openai` - Needs OPENAI_API_KEY
- `anthropic` - Needs ANTHROPIC_API_KEY
- `huggingface` - Optional HUGGINGFACE_API_KEY

---

## 🎓 **For Your Thesis**

### **Recommended Setup:**

```bash
# 1. Use CASCADE for quick tests (FREE)
scitrans test --backend cascade

# 2. Get DeepSeek API key for experiments ($0.14/1M tokens)
export DEEPSEEK_API_KEY="your-key"
scitrans test --backend deepseek

# 3. Optional: Get OpenAI for final comparisons ($2.50/1M tokens)
export OPENAI_API_KEY="sk-your-key"
scitrans test --backend openai

# This gives you:
# - FREE option for testing
# - CHEAP option for bulk experiments
# - BEST option for final results
```

### **Budget Estimation:**

For 100 papers (2000 pages total):
- **CASCADE:** $0
- **DeepSeek:** ~$0.40
- **GPT-4o-mini:** ~$0.80
- **GPT-4o:** ~$12.50
- **Claude 3.5:** ~$18.00

**Recommendation:** Use CASCADE + DeepSeek for entire thesis work = **$0.40 total!** 💰

---

## ✅ **Ready to Use!**

**Quick Start:**
```bash
# 1. Test free backend
scitrans test --backend cascade

# 2. If satisfied, use it!
scitrans translate paper.pdf --backend cascade

# 3. Need better quality? Get DeepSeek key (cheap!)
export DEEPSEEK_API_KEY="your-key"
scitrans translate paper.pdf --backend deepseek

# 4. Done! 🎉
```

**All backends are now ready to use!** 🚀
