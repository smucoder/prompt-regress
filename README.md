# prompt-regress 🔍

**AI Model Output Regression Testing Tool**

When upgrading prompts or switching models (e.g. GPT-4 → Claude Opus), developers need a quick way to know if outputs broke. `prompt-regress` solves this by comparing model outputs across prompt versions or model versions.

## 🚀 Features

- **Model-Agnostic**: Works with OpenAI, Anthropic, local models (Ollama), and more
- **Semantic Similarity**: Beyond text matching - understands meaning changes
- **Cost Tracking**: Monitor token usage and cost differences
- **JSON Validation**: Ensure structured outputs remain valid

## 📦 Installation

```bash
pip install prompt-regress
```

## 🏃 Quick Start

### 1. Initialize Configuration

```bash
prompt-regress init
```

This creates a `prompt-regress.yml` configuration file:

```yaml
metrics:
  semantic_similarity:
    threshold: 0.8
  text_similarity:
    threshold: 0.7
models:
- name: gpt-4
  parameters:
    max_tokens: 1000
    temperature: 0.7
  provider: openai
- name: claude-4
  parameters:
    max_tokens: 1000
    temperature: 0.7
  provider: anthropic
- host: http://localhost:11434
  name: deepseek-r1:1.5b
  parameters:
    max_tokens: 1000
    temperature: 0.7
  provider: local
test_cases:
- inputs:
  - text: Sample text to summarize
  name: summarization
  prompt_template: 'Summarize this text in 2-3 sentences: {text}'
- inputs:
  - context: This is a sample context.
    question: What is the context about?
  - context: My name is Foo.
    question: What is my name?
  name: question_answering
  prompt_template: 'Answer the question based on the context: {context} Question:
    {question}'
- expect_json: true
  inputs:
  - data: John Doe, age 30, works at TechCorp
  name: json_extraction
  prompt_template: 'Extract key information as JSON: {data}'
```

### 2. Compare Models

```bash
# Compare two models
prompt-regress check --baseline gpt-4 --target claude-opus

# Output example:
🔍 Prompt Regression Test Report
==================================================
✅ Passed: 2/2
❌ Failed: 0/2

✅ PASS summarization
  Text Similarity: 0.856
  Semantic Similarity: 0.923
  Token Difference: +12
  Cost Difference: +$0.000180

✅ PASS json_extraction
  Text Similarity: 0.734
  Semantic Similarity: 0.891
  Token Difference: -5
  Cost Difference: -$0.000075
```

### 3. CI/CD Integration

```bash
# Fail build if regressions detected
prompt-regress check \
  --baseline gpt-4 \
  --target claude-opus \
  --fail-on-regression
```

## 🔧 CLI Commands

### Initialize Project
```bash
prompt-regress init [--config prompt-regress.yml]
```

### Compare Models
```bash
prompt-regress check \
  --baseline MODEL_NAME \
  --target MODEL_NAME \
  [--config CONFIG_FILE] \
  [--format console|json] \
  [--fail-on-regression]
```

### List Available Models
```bash
prompt-regress models [--config prompt-regress.yml]
```

### List Test Cases
```bash
prompt-regress tests [--config prompt-regress.yml]
```

## 🐍 Python SDK

```python
from prompt_regress import PromptRegress

# Initialize
regress = PromptRegress("prompt-regress.yml")

# Compare models
results = regress.compare_models("gpt-4", "claude-opus")

# Generate report
report = regress.generate_report(results, format="json")
print(report)

# Check individual results
for result in results:
    if not result.passed:
        print(f"❌ {result.test_case} failed!")
        print(f"   Semantic similarity: {result.semantic_similarity:.3f}")
```

## 🤖 Supported Providers

| Provider | Models | API Key Required |
|----------|--------|--------------|
| OpenAI | gpt-4, gpt-3.5-turbo, etc. | ✅ |
| Anthropic | claude-opus, claude-sonnet | ✅ |
| Local (Ollama) | llama2, codellama, etc. | ❌ |


## 📊 Comparison Metrics

- **Text Similarity**: Exact text matching using difflib
- **Semantic Similarity**: Meaning comparison using sentence transformers
- **Token Usage**: Track token consumption changes
- **Cost Analysis**: Monitor API cost differences
- **JSON Validation**: Ensure structured outputs remain valid
- **Performance**: Response time and throughput

## 🚦 GitHub Actions Integration

Add to `.github/workflows/prompt-regression.yml`:

```yaml
name: Prompt Regression Tests

on:
  pull_request:
    paths: ['prompts/**', 'prompt-regress.yml']

jobs:
  regression-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install prompt-regress
      run: pip install prompt-regress
    
    - name: Run regression tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      run: |
        prompt-regress check \
          --baseline gpt-4 \
          --target claude-opus \
          --fail-on-regression
```

## 🎯 Use Cases

### 1. Model Migration
```bash
# Switching from GPT-4 to Claude Opus
prompt-regress check --baseline gpt-4 --target claude-opus
```

### 2. Prompt Optimization
```bash
# Test prompt changes with same model
prompt-regress check --baseline gpt-4-v1 --target gpt-4-v2
```

### 3. Cost Optimization
```bash
# Compare expensive vs cheaper models
prompt-regress check --baseline gpt-4 --target gpt-3.5-turbo
```

### 4. Local Model Testing
```bash
# Compare cloud vs local models
prompt-regress check --baseline gpt-4 --target llama2-local
```

## ⚙️ Configuration

### Model Configuration
```yaml
models:
  - name: custom-gpt-4
    provider: openai
    parameters:
      temperature: 0.5
      max_tokens: 2000
    
  - name: local-llama
    provider: local
    host: http://localhost:11434
    model: llama2
```

### Test Case Configuration
```yaml
test_cases:
  - name: code_generation
    prompt: "Generate Python code for: {task}"
    inputs:
      - task: "sort a list of dictionaries by key"
      - task: "create a REST API endpoint"
    expect_json: false
    timeout: 30
    
  - name: data_extraction
    prompt: "Extract data as JSON: {text}"
    inputs:
      - text: "Company: Acme Corp, Revenue: $1M, Employees: 50"
    expect_json: true
```

### Metrics Configuration
```yaml
metrics:
  semantic_similarity:        # Minimum semantic similarity (0-1)
    threshold: 0.8             
  text_similarity:            # Minimum text similarity (0-1)
    threshold: 0.7
```

## 🧪 Advanced Usage

### Custom Similarity Functions
```python
from prompt_regress import PromptRegress

class CustomPromptRegress(PromptRegress):
    def _calculate_custom_similarity(self, text1: str, text2: str) -> float:
        # Your custom similarity logic here
        return similarity_score
```

### Custom Providers
```python
from prompt_regress import ModelProvider

class CustomProvider(ModelProvider):
    def generate(self, prompt: str) -> Tuple[str, int]:
        # Your custom model API integration
        response = your_api_call(prompt)
        return response.text, response.token_count
```

## 📈 Monitoring & Alerts

### Slack Integration
```bash
# Send results to Slack webhook
prompt-regress check \
  --baseline gpt-4 \
  --target claude-opus \
  --format json | \
  curl -X POST -H 'Content-type: application/json' \
  --data @- $SLACK_WEBHOOK_URL
```

### Email Alerts
```python
import smtplib
from prompt_regress import PromptRegress

regress = PromptRegress()
results = regress.compare_models("gpt-4", "claude-opus")

failed_tests = [r for r in results if not r.passed]
if failed_tests:
    send_email_alert(f"Regression detected in {len(failed_tests)} tests")
```

## 🔒 Security

### API Key Management
```bash
# Environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Or use .env file
echo "OPENAI_API_KEY=your-key-here" >> .env
echo "ANTHROPIC_API_KEY=your-key-here" >> .env
```

### Rate Limiting
```yaml
models:
  - name: gpt-4
    provider: openai
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 40000
```

## 🐛 Troubleshooting

### Common Issues

**1. Sentence Transformers Download**
```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**2. API Key Issues**
```bash
# Test API connectivity
prompt-regress models --config test-config.yml
```

**3. Memory Issues with Large Inputs**
```yaml
# Reduce batch size
batch_size: 1
max_input_length: 2000
```

### Debug Mode
```bash
# Enable verbose logging
prompt-regress check --baseline gpt-4 --target claude-opus --verbose
```
## 🏆 Why prompt-regress?

### Before prompt-regress:
- ❌ Manual testing of prompt changes
- ❌ No visibility into model output quality
- ❌ Expensive mistakes in production
- ❌ Time-consuming model comparisons

### After prompt-regress:
- ✅ Automated regression testing
- ✅ Quantified quality metrics
- ✅ Catch issues before deployment
- ✅ Efficient model evaluation

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for semantic similarity
- [Click](https://github.com/pallets/click) for the CLI framework
- The AI community for inspiration and feedback

---

⭐ **Star this repo if prompt-regress helps you build better AI applications!**