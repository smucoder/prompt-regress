# 🧪🧪🧪 Regression Examples

## Basic

```yaml
models:
- name: gpt-4
  provider: openai
  parameters:
    max_tokens: 1000
    temperature: 0.7
- name: claude-4
  provider: anthropic
  parameters:
    max_tokens: 1000
    temperature: 0.7
- name: deepseek-r1:1.5b
  host: http://localhost:11434
  provider: local
  parameters:
    max_tokens: 1000
    temperature: 0.7
test_cases:
- name: summarization
  prompt_template: 'Summarize this text in 2-3 sentences: {text}'
  inputs:
  - text: Sample text to summarize
- name: question_answering
  prompt_template: 'Answer the question based on the context: {context} Question:
    {question}'
  inputs:
  - context: This is a sample context.
    question: What is the context about?
  - context: My name is Foo.
    question: What is my name?
metrics:
  semantic_similarity:
    threshold: 0.8
  text_similarity:
    threshold: 0.7
```

## Prompt Variables in test cases
* Support defining prompt variables in prompt template. Values to the prompt varaibles must be provided in the inputs section with keys matching the prompt variable names
```yaml
test_cases:
- name: question_answering
  prompt_template: 'Answer the question based on the context: {context} Question:
    {question}'
  inputs:
  - context: This is a sample context.
    question: What is the context about?
  - context: My name is Foo.
    question: What is my name?
```

## Specifying Regression Test Settings (Optional)
```yaml
regression_options:
  max_concurrency: 5
  embedding_model: Qwen/Qwen3-Embedding-0.6B
```

## Structured Output Test Case
* Currently only supported Json Validation
```yaml
test_cases:
- name: json_extraction
  prompt_template: 'Extract key information as JSON: {data}'
  inputs:
  - data: John Doe, age 30, works at TechCorp
  expect_json: true
``` 

## Multiline Prompt and Input Example

```yaml
test_cases:
- name: summarization
  prompt_template: 'Summarize the given data: {data}'
  inputs:
  - data: |
      The story begins in a small village by the sea.
      A young sailor dreamed of distant lands.
      His adventure would change everything.
  - data: |
      The story begins in a small village by the sea.
      A young sailor dreamed of distant lands.
      His adventure would change everything.
```

## Full Example
```yaml
models:
- name: gpt-4
  provider: openai
  parameters:
    max_tokens: 1000
    temperature: 0.7
- name: claude-4
  provider: anthropic
  parameters:
    max_tokens: 1000
    temperature: 0.7
- name: deepseek-r1:1.5b
  host: http://localhost:11434
  provider: local
  parameters:
    max_tokens: 1000
    temperature: 0.7
test_cases:
- name: summarization
  prompt_template: 'Summarize this text in 2-3 sentences: {text}'
  inputs:
  - text: Sample text to summarize
- name: question_answering
  prompt_template: 'Answer the question based on the context: {context} Question:
    {question}'
  inputs:
  - context: This is a sample context.
    question: What is the context about?
  - context: My name is Foo.
    question: What is my name?
- name: json_extraction
  prompt_template: 'Extract key information as JSON: {data}'
  inputs:
  - data: John Doe, age 30, works at TechCorp
  expect_json: true
metrics:
  semantic_similarity:
    threshold: 0.8
  text_similarity:
    threshold: 0.7
regression_options:
  max_concurrency: 5
```
