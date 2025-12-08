#!/bin/bash
# Setup script for SciTrans-LLMs NEW

echo "🚀 Setting up SciTrans-LLMs NEW..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✓ Python $python_version is installed${NC}"
else
    echo -e "${RED}✗ Python $required_version or higher is required. Current: $python_version${NC}"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt --quiet

# Create necessary directories
echo "Creating project directories..."
mkdir -p corpus/{training,validation,test}
mkdir -p thesis/{data,figures,tables,results}
mkdir -p configs
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"

# Download sample data (optional)
echo "Do you want to download sample data? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Downloading sample data..."
    # wget or curl commands to download sample PDFs
    echo -e "${YELLOW}Sample data download not yet implemented${NC}"
fi

# Set up default configuration
if [ ! -f "configs/default.yaml" ]; then
    echo "Creating default configuration..."
    cat > configs/default.yaml << EOF
# Default configuration for SciTrans-LLMs NEW
source_lang: en
target_lang: fr
backend: openai
model_name: gpt-4o
num_candidates: 3
enable_masking: true
enable_context: true
preserve_layout: true
enable_glossary: true
domain: scientific
quality_threshold: 0.7
batch_size: 10
cache_enabled: true
EOF
    echo -e "${GREEN}✓ Default configuration created${NC}"
fi

# Check for API keys
echo ""
echo "Checking for API keys..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠ OPENAI_API_KEY not set${NC}"
fi
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}⚠ ANTHROPIC_API_KEY not set${NC}"
fi
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo -e "${YELLOW}⚠ DEEPSEEK_API_KEY not set${NC}"
fi

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Set your API keys:"
echo "   export OPENAI_API_KEY='your-key-here'"
echo "   export ANTHROPIC_API_KEY='your-key-here'"
echo "   export DEEPSEEK_API_KEY='your-key-here'"
echo ""
echo "2. Launch the GUI:"
echo "   python gui/app.py"
echo ""
echo "3. Or use the CLI:"
echo "   python -m scitran translate --help"
echo ""
echo "4. Run tests:"
echo "   pytest tests/"
echo ""
echo "Happy translating! 🎉"
