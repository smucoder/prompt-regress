import yaml
import json
import asyncio

from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, asdict
from .models import OpenAIProvider, AnthropicProvider, LocalProvider, ModelProvider
from .metrics import SimilarityMetrics

@dataclass
class ComparisonResult:
    """Class to hold the results of a model comparison."""
    test_case: str
    prompt: str
    baseline_output: str
    target_output: str
    text_similarity: float  
    semantic_similarity: float
    passed: bool

class PromptRegress:
    def __init__(self, config_path: Path):
        """
        Initialize the Prompt Regress instance with a configuration file.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.regression_options = self.config.get('regression_options', {})
        self.max_concurrency = self.regression_options.get('max_concurrency', 5)
        embedding_model = self.regression_options.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B')
        self.metrics = SimilarityMetrics(embedding_model=embedding_model)

    def load_config(self):
        """
        Load the configuration from the yaml file.

        Returns:
            dict: Configuration data.
        """
        if not self.config_path.exists():
            return self._create_default_config()
        
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
        
    def _create_default_config(self):
        """Create a default configuration."""
        default_config = {
            'models': [
                {
                    'name': 'gpt-4',
                    'provider': 'openai',
                    'parameters':{
                        'temperature': 0.7,
                        'max_tokens': 1000
                    }

                },
                {
                    'name': 'claude-4',
                    'provider': 'anthropic',
                    'parameters':{
                        'temperature': 0.7,
                        'max_tokens': 1000
                    }
                },
                {
                    'name': 'deepseek-r1:1.5b',
                    'provider': 'local',
                    'host': 'http://localhost:11434',
                    'parameters':{
                        'temperature': 0.7,
                        'max_tokens': 1000
                    }
                }
            ],
            'test_cases': [
                {
                    'name': 'summarization',
                    'prompt_template': 'Summarize this text in 2-3 sentences: {text}',
                    'inputs': [{'text': 'Sample text to summarize'}],
                },
                {
                    'name': 'question_answering',
                    'prompt_template': 'Answer the question based on the context: {context} Question: {question}',
                    'inputs': [{'context': 'This is a sample context.', 'question': 'What is the context about?'}, {'context': 'My name is Foo.', 'question': 'What is my name?'}],
                },
                {
                    'name': 'json_extraction',
                    'prompt_template': 'Extract key information as JSON: {data}',
                    'inputs': [{'data': 'John Doe, age 30, works at TechCorp'}],
                    'expect_json': True
                }
            ],
            'metrics': {
                'text_similarity': {'threshold': 0.7},
                'semantic_similarity': {'threshold': 0.8}
            },
            'regression_options': {
                'max_concurrency': 5
            }
        }
        
        # Save default config
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def _is_valid_json(result):
        """
        Check whether given text is a valid json

        Args:
            result: text

        Returns:
            True or False
        """
        try:
            json.loads(result)
            return True
        except json.JSONDecodeError:
            return False
    
    def _get_provider(self, model_config: Dict[str, Any]) -> ModelProvider:
        """
        Get the provider instance based on the model configuration.

        Args:
            model_config (Dict[str, Any]): Model configuration containing provider information.

        Returns:
            ModelProvider: An instance of the provider class.
        """

        provider_name = model_config['provider']
        
        if provider_name == 'openai':
            provider = OpenAIProvider(model=model_config['name'], max_concurrency=self.max_concurrency)
        elif provider_name == 'anthropic':
            provider = AnthropicProvider(max_concurrency=self.max_concurrency)
        elif provider_name == 'local':
            if 'host' not in model_config:
                print("⚠️ Specified Local provider but 'host' not provided in model configuration. Using default 'http://localhost:11434'.")
            provider = LocalProvider(host=model_config.get('host', "http://localhost:11434"), max_concurrency=self.max_concurrency)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        return provider
    
    async def arun_test_case(self, test_case: dict, model_config: Dict[str, Any]):
        """
        Asynchronously run a test case against a specified model.

        Args:
            test_case (dict): Test case configuration.
            model_config (Dict[str, Any]): Model configuration.

        Returns:
            dict: Results of the test case execution.
        """
        provider = self._get_provider(model_config)

        prompts = []
        for input_data in test_case['inputs']:
            prompt = test_case['prompt_template'].format(**input_data)
            prompts.append(prompt)

        tasks = []
        for prompt in prompts:
            task = provider.agenerate(prompt, model=model_config['name'], **model_config.get('parameters', {}))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results
      
    async def acompare_models(self, baseline: str, target: str) -> List[ComparisonResult]:
        """
        Compare outputs between two models.

        Args:
            baseline (str): Baseline model name.
            target (str): Target model name.

        Returns:
            dict: Comparison results.
        """
        baseline_config = next((m for m in self.config['models'] if m['name'] == baseline), None)
        target_config = next((m for m in self.config['models'] if m['name'] == target), None)
        if baseline_config is None or target_config is None:
            raise ValueError(f"⚠️ One or both models not found in configuration. Provided models: baseline={baseline}, target={target}")
        
        if 'metrics' not in self.config:
            print("⚠️ Metrics are missing in the configuration. Using default metrics text and semantic similarity " \
            "with thresholds 0.7 and 0.8.")
            self.config['metrics'] = {
                'text_similarity': {'threshold': 0.7},
                'semantic_similarity': {'threshold': 0.8}
            }

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_with_semaphore(test_case, model_config):
            async with semaphore:
                return await self.arun_test_case(test_case, model_config)

        test_cases = self.config.get('test_cases', [])

        tasks = []
        for test_case in test_cases:
            baseline_task = run_with_semaphore(test_case, baseline_config)
            target_task = run_with_semaphore(test_case, target_config)
            tasks.extend([baseline_task, target_task])

        all_results = await asyncio.gather(*tasks)

        baseline_results_list = all_results[::2]
        target_results_list = all_results[1::2]

        results = []
        for i, test_case in enumerate(test_cases):
            baseline_results = baseline_results_list[i]
            target_results = target_results_list[i]

            for baseline_result, target_result in zip(baseline_results, target_results):
                metric_results = []
                if 'text_similarity' in self.config['metrics']:
                    text_sim = self.metrics.text_similarity(baseline_result.text, target_result.text)
                    metric_results.append(text_sim >= self.config['metrics']['text_similarity']['threshold'])
                if 'semantic_similarity' in self.config['metrics']:
                    semantic_sim = self.metrics.semantic_similarity(baseline_result.text, target_result.text)
                    metric_results.append(semantic_sim >= self.config['metrics']['semantic_similarity']['threshold'])
                if test_case.get('expect_json', False):
                    is_baseline_valid_json = self._is_valid_json(baseline_result.text)
                    is_target_valid_json = self._is_valid_json(target_result.text)
                    metric_results.extend([is_baseline_valid_json, is_target_valid_json])

                passed = all(metric_results) if metric_results else False
                
                result = ComparisonResult(
                    test_case=test_case['name'],
                    prompt=baseline_result.prompt,
                    baseline_output=baseline_result.text,
                    target_output=target_result.text,
                    text_similarity=text_sim,
                    semantic_similarity=semantic_sim,
                    passed=passed
                )

                results.append(result)
        return results
                            
    def generate_report(self, results, verbose, format='console') -> str:
        if format == 'json':
            return json.dumps([asdict(r) for r in results], indent=2)
        
        elif format == 'console':
            report = []
            report.append("🔍 Prompt Regression Test Report")
            report.append("=" * 50)
            
            passed_count = sum(1 for r in results if r.passed)
            total_count = len(results)
            
            report.append(f"✅ Passed: {passed_count}/{total_count}")
            report.append(f"❌ Failed: {total_count - passed_count}/{total_count}")
            
            report.append("")
            
            for result in results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                report.append(f"{status} {result.test_case}")
                if verbose:
                    report.append(f"  Prompt: {result.prompt}")
                    report.append(f"  Baseline Output: {result.baseline_output}")
                    report.append(f"  Target Output: {result.target_output}")
                report.append(f"  Text Similarity: {result.text_similarity:.3f}")
                report.append(f"  Semantic Similarity: {result.semantic_similarity:.3f}")
                
                report.append("")
            
            return "\n".join(report)
        
        else:
            raise ValueError(f"Unknown output format: {format}")
                
                    

if __name__ == "__main__":
    regress = PromptRegress(Path('prompt-regress.yml'))
