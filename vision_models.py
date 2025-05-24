import openai
import anthropic
import base64
import json
import time
import asyncio
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class ModelResponse:
    """Structure for model responses."""
    model_name: str
    perceptual_choice: int  # 1 or 2 for first/second interval
    confidence: int  # 1-6 scale
    response_time: float
    raw_response: str
    trial_id: str
    stimulus_info: Dict

class VisionModelInterface:
    """
    Interface for testing vision language models on metacognition tasks.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 anthropic_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None):
        """
        Initialize API clients.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4o
            anthropic_api_key: Anthropic API key for Claude Sonnet 3.5
            gemini_api_key: Google API key for Gemini Pro Vision
        """
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None
        
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        if anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_client = genai.GenerativeModel('gemini-1.5-pro')
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_prompt(self, trial_type="2afc"):
        """
        Create the instruction prompt for the vision models.
        
        Args:
            trial_type: Type of trial ("2afc" for two-alternative forced choice)
        """
        if trial_type == "2afc":
            return """You are participating in a perceptual decision-making experiment. You will be shown two images representing temporal intervals containing visual stimuli.

TASK:
1. Look at both images carefully
2. Identify which temporal interval (FIRST or SECOND) contains the Gabor patch with HIGHER CONTRAST
3. Rate your confidence in this decision on a scale of 1-6

Each image shows 6 Gabor patches arranged around a central fixation point. One patch will have higher contrast than the others.

RESPONSE FORMAT:
Please respond with exactly this format:
CHOICE: [1 or 2] (1 for first interval, 2 for second interval)
CONFIDENCE: [1-6] (1 = very low confidence, 6 = very high confidence)
REASONING: [brief explanation of your decision]

Use the full confidence scale:
- 1: Very low confidence, mostly guessing
- 2: Low confidence, uncertain
- 3: Somewhat low confidence
- 4: Somewhat high confidence  
- 5: High confidence, quite sure
- 6: Very high confidence, extremely sure

Now examine the two temporal intervals:"""
    
    async def query_gpt4o(self, first_image_b64: str, second_image_b64: str, 
                         trial_info: Dict, model_name: str = "gpt-4o") -> ModelResponse:
        """
        Query GPT-4o with vision capability.
        
        Args:
            first_image_b64: Base64 encoded first interval image
            second_image_b64: Base64 encoded second interval image
            trial_info: Information about the trial
            model_name: Specific GPT model version to use
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")
        
        prompt = self.create_prompt()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "\nFIRST TEMPORAL INTERVAL:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{first_image_b64}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": "\nSECOND TEMPORAL INTERVAL:"},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{second_image_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        start_time = time.time()
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.1
            )
            
            response_time = time.time() - start_time
            raw_response = response.choices[0].message.content
            
            # Parse response
            choice, confidence = self._parse_response(raw_response)
            
            return ModelResponse(
                model_name=model_name,
                perceptual_choice=choice,
                confidence=confidence,
                response_time=response_time,
                raw_response=raw_response,
                trial_id=trial_info.get('trial_id', ''),
                stimulus_info=trial_info
            )
            
        except Exception as e:
            self.logger.error(f"GPT query failed with model {model_name}: {e}")
            return ModelResponse(
                model_name=model_name,
                perceptual_choice=-1,  # Error indicator
                confidence=-1,
                response_time=time.time() - start_time,
                raw_response=f"Error: {str(e)}",
                trial_id=trial_info.get('trial_id', ''),
                stimulus_info=trial_info
            )
    
    async def query_claude(self, first_image_b64: str, second_image_b64: str,
                          trial_info: Dict, model_name: str = "claude-3-5-sonnet-20241022") -> ModelResponse:
        """
        Query Claude Sonnet 3.5 with vision capability.
        
        Args:
            first_image_b64: Base64 encoded first interval image
            second_image_b64: Base64 encoded second interval image
            trial_info: Information about the trial
            model_name: Specific Claude model version to use
        """
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized. Please provide API key.")
        
        prompt = self.create_prompt()
        
        start_time = time.time()
        
        try:
            message = self.anthropic_client.messages.create(
                model=model_name,
                max_tokens=500,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "\nFIRST TEMPORAL INTERVAL:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": first_image_b64
                                }
                            },
                            {"type": "text", "text": "\nSECOND TEMPORAL INTERVAL:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64", 
                                    "media_type": "image/png",
                                    "data": second_image_b64
                                }
                            }
                        ]
                    }
                ]
            )
            
            response_time = time.time() - start_time
            raw_response = message.content[0].text
            
            # Parse response
            choice, confidence = self._parse_response(raw_response)
            
            return ModelResponse(
                model_name=model_name,
                perceptual_choice=choice,
                confidence=confidence,
                response_time=response_time,
                raw_response=raw_response,
                trial_id=trial_info.get('trial_id', ''),
                stimulus_info=trial_info
            )
            
        except Exception as e:
            self.logger.error(f"Claude query failed with model {model_name}: {e}")
            return ModelResponse(
                model_name=model_name,
                perceptual_choice=-1,  # Error indicator
                confidence=-1,
                response_time=time.time() - start_time,
                raw_response=f"Error: {str(e)}",
                trial_id=trial_info.get('trial_id', ''),
                stimulus_info=trial_info
            )
    
    def _parse_response(self, raw_response: str) -> Tuple[int, int]:
        """
        Parse model response to extract choice and confidence.
        
        Args:
            raw_response: Raw text response from model
            
        Returns:
            Tuple of (choice, confidence)
        """
        choice = -1
        confidence = -1
        
        lines = raw_response.strip().split('\n')
        
        for line in lines:
            line = line.strip().upper()
            
            # Look for choice
            if 'CHOICE:' in line:
                try:
                    choice_part = line.split('CHOICE:')[1].strip()
                    choice = int(choice_part.split()[0])
                    if choice not in [1, 2]:
                        choice = -1
                except (ValueError, IndexError):
                    pass
            
            # Look for confidence
            if 'CONFIDENCE:' in line:
                try:
                    conf_part = line.split('CONFIDENCE:')[1].strip()
                    confidence = int(conf_part.split()[0])
                    if confidence not in range(1, 7):
                        confidence = -1
                except (ValueError, IndexError):
                    pass
        
        # Fallback parsing if structured format not found
        if choice == -1 or confidence == -1:
            import re
            
            # Look for numbers that could be choice/confidence
            numbers = re.findall(r'\b([1-6])\b', raw_response)
            
            if len(numbers) >= 2 and choice == -1:
                potential_choice = int(numbers[0])
                if potential_choice in [1, 2]:
                    choice = potential_choice
            
            if len(numbers) >= 2 and confidence == -1:
                potential_conf = int(numbers[1]) if len(numbers) > 1 else int(numbers[0])
                if potential_conf in range(1, 7):
                    confidence = potential_conf
        
        return choice, confidence
    
    async def run_trial(self, first_image_b64: str, second_image_b64: str,
                       trial_info: Dict, models: List[str] = None) -> List[ModelResponse]:
        """
        Run a trial with specified models.
        
        Args:
            first_image_b64: Base64 encoded first interval image
            second_image_b64: Base64 encoded second interval image
            trial_info: Information about the trial
            models: List of models to test (e.g., ["gpt-4o", "claude"] or specific versions)
            
        Returns:
            List of ModelResponse objects
        """
        if models is None:
            models = []
            if self.openai_client:
                models.append("gpt-4o")
            if self.anthropic_client:
                models.append("claude")
        
        tasks = []
        
        for model in models:
            # More flexible model matching - check if model name contains the base model name
            if ("gpt-4o" in model.lower() or model.lower() == "gpt-4o") and self.openai_client:
                task = self.query_gpt4o(first_image_b64, second_image_b64, trial_info, model)
                tasks.append(task)
            elif ("claude" in model.lower() or model.lower() == "claude") and self.anthropic_client:
                task = self.query_claude(first_image_b64, second_image_b64, trial_info, model)
                tasks.append(task)
        
        if not tasks:
            raise ValueError("No valid models available for testing")
        
        # Run all model queries concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = []
        for response in responses:
            if isinstance(response, ModelResponse):
                valid_responses.append(response)
            else:
                self.logger.error(f"Model query failed: {response}")
        
        return valid_responses

class MetacognitionAnalyzer:
    """
    Analyzes metacognitive performance of vision models.
    """
    
    def __init__(self):
        self.responses = []
    
    def add_response(self, response: ModelResponse, correct_answer: int):
        """
        Add a response with ground truth for analysis.
        
        Args:
            response: ModelResponse object
            correct_answer: Correct choice (1 or 2)
        """
        response_data = {
            'model': response.model_name,
            'trial_id': response.trial_id,
            'choice': response.perceptual_choice,
            'confidence': response.confidence,
            'correct': response.perceptual_choice == correct_answer,
            'correct_answer': correct_answer,
            'response_time': response.response_time,
            'stimulus_info': response.stimulus_info
        }
        self.responses.append(response_data)
    
    def calculate_performance_metrics(self, model_name: str = None) -> Dict:
        """
        Calculate performance and metacognitive metrics.
        
        Args:
            model_name: Specific model to analyze (None for all)
        """
        if model_name:
            data = [r for r in self.responses if r['model'] == model_name]
        else:
            data = self.responses
        
        if not data:
            return {}
        
        # Basic performance
        accuracy = sum(r['correct'] for r in data) / len(data)
        
        # Confidence analysis
        confidences = [r['confidence'] for r in data if r['confidence'] > 0]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Metacognitive sensitivity (confidence difference between correct/incorrect)
        correct_confidences = [r['confidence'] for r in data if r['correct'] and r['confidence'] > 0]
        incorrect_confidences = [r['confidence'] for r in data if not r['correct'] and r['confidence'] > 0]
        
        meta_sensitivity = 0
        if correct_confidences and incorrect_confidences:
            mean_correct_conf = sum(correct_confidences) / len(correct_confidences)
            mean_incorrect_conf = sum(incorrect_confidences) / len(incorrect_confidences)
            meta_sensitivity = mean_correct_conf - mean_incorrect_conf
        
        # Response time analysis
        response_times = [r['response_time'] for r in data if r['response_time'] > 0]
        mean_rt = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'accuracy': accuracy,
            'n_trials': len(data),
            'mean_confidence': mean_confidence,
            'metacognitive_sensitivity': meta_sensitivity,
            'mean_response_time': mean_rt,
            'confidence_distribution': self._get_confidence_distribution(data),
            'accuracy_by_confidence': self._get_accuracy_by_confidence(data)
        }
    
    def _get_confidence_distribution(self, data: List[Dict]) -> Dict:
        """Get distribution of confidence ratings."""
        dist = {i: 0 for i in range(1, 7)}
        for r in data:
            if 1 <= r['confidence'] <= 6:
                dist[r['confidence']] += 1
        return dist
    
    def _get_accuracy_by_confidence(self, data: List[Dict]) -> Dict:
        """Get accuracy for each confidence level."""
        acc_by_conf = {}
        for conf in range(1, 7):
            conf_trials = [r for r in data if r['confidence'] == conf]
            if conf_trials:
                accuracy = sum(r['correct'] for r in conf_trials) / len(conf_trials)
                acc_by_conf[conf] = {'accuracy': accuracy, 'n_trials': len(conf_trials)}
        return acc_by_conf
    
    def get_summary_report(self) -> str:
        """Generate a summary report of all models."""
        report = "METACOGNITION EXPERIMENT SUMMARY\n" + "="*50 + "\n\n"
        
        models = list(set(r['model'] for r in self.responses))
        
        for model in models:
            metrics = self.calculate_performance_metrics(model)
            
            report += f"MODEL: {model}\n"
            report += f"Trials: {metrics.get('n_trials', 0)}\n"
            report += f"Accuracy: {metrics.get('accuracy', 0):.3f}\n"
            report += f"Mean Confidence: {metrics.get('mean_confidence', 0):.2f}\n"
            report += f"Metacognitive Sensitivity: {metrics.get('metacognitive_sensitivity', 0):.3f}\n"
            report += f"Mean Response Time: {metrics.get('mean_response_time', 0):.2f}s\n"
            report += "\nConfidence Distribution:\n"
            
            conf_dist = metrics.get('confidence_distribution', {})
            for conf, count in conf_dist.items():
                report += f"  {conf}: {count} trials\n"
            
            report += "\nAccuracy by Confidence:\n"
            acc_by_conf = metrics.get('accuracy_by_confidence', {})
            for conf, data in acc_by_conf.items():
                report += f"  {conf}: {data['accuracy']:.3f} ({data['n_trials']} trials)\n"
            
            report += "\n" + "-"*30 + "\n\n"
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # This would normally be run with actual API keys
    print("Vision Models Interface Module")
    print("To use: provide OpenAI and/or Anthropic API keys")
    
    # Example of how to initialize
    # interface = VisionModelInterface(
    #     openai_api_key="your-openai-key",
    #     anthropic_api_key="your-anthropic-key"
    # ) 