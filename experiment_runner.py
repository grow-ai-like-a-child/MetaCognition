import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import os
from pathlib import Path
import openai
import anthropic

from gabor_stimulus import GaborStimulus
from staircase import AdaptiveStaircase, MultiLocationStaircase
from vision_models import VisionModelInterface, MetacognitionAnalyzer, ModelResponse

class MetacognitionExperiment:
    """
    Main experiment runner for the metacognition investigation.
    Coordinates stimulus generation, staircase procedures, and model testing.
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 output_dir: str = "experiment_results",
                 n_trials_per_session: int = 50,
                 target_performance: float = 0.71):
        """
        Initialize the experiment.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4o
            anthropic_api_key: Anthropic API key for Claude Sonnet 3.5
            output_dir: Directory to save results
            n_trials_per_session: Number of trials per experimental session
            target_performance: Target performance level for staircase
        """
        # Initialize components
        self.stimulus_generator = GaborStimulus()
        self.staircase = AdaptiveStaircase(target_performance=target_performance)
        self.vision_interface = VisionModelInterface(openai_api_key, anthropic_api_key)
        self.analyzer = MetacognitionAnalyzer()
        
        # Experiment parameters
        self.n_trials_per_session = n_trials_per_session
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Trial tracking
        self.current_trial = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / f"experiment_{self.session_id}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized metacognition experiment - Session ID: {self.session_id}")
    
    def generate_trial_stimulus(self, trial_number: int) -> Dict:
        """
        Generate stimulus for a single trial.
        
        Args:
            trial_number: Current trial number
            
        Returns:
            Dictionary containing trial information and stimuli
        """
        # Get current contrast from staircase
        target_contrast = self.staircase.get_current_contrast()
        
        # Randomly select locations and which interval has higher contrast
        first_location = np.random.randint(0, 6)
        second_location = np.random.randint(0, 6)
        
        # Ensure different locations
        while second_location == first_location:
            second_location = np.random.randint(0, 6)
        
        # Randomly decide which interval has the higher contrast target
        higher_contrast_interval = np.random.choice([1, 2])
        
        if higher_contrast_interval == 1:
            first_contrast = target_contrast
            second_contrast = target_contrast * 0.7  # Lower contrast
        else:
            first_contrast = target_contrast * 0.7  # Lower contrast
            second_contrast = target_contrast
        
        # Generate the two temporal intervals
        first_image, second_image = self.stimulus_generator.create_temporal_sequence(
            first_target_loc=first_location,
            second_target_loc=second_location,
            first_contrast=first_contrast,
            second_contrast=second_contrast
        )
        
        # Convert to base64 for API transmission
        first_image_b64 = self.stimulus_generator.image_to_base64(first_image)
        second_image_b64 = self.stimulus_generator.image_to_base64(second_image)
        
        trial_info = {
            'trial_id': f"{self.session_id}_trial_{trial_number:03d}",
            'trial_number': trial_number,
            'session_id': self.session_id,
            'first_location': first_location,
            'second_location': second_location,
            'first_contrast': first_contrast,
            'second_contrast': second_contrast,
            'target_contrast': target_contrast,
            'correct_answer': higher_contrast_interval,
            'first_image_b64': first_image_b64,
            'second_image_b64': second_image_b64,
            'timestamp': datetime.now().isoformat()
        }
        
        return trial_info
    
    async def run_single_trial(self, trial_number: int, session_dir: Path, models: List[str] = None):
        """
        Run a single trial with specified models.
        
        Args:
            trial_number: Current trial number
            session_dir: Session directory for saving data
            models: List of models to test
            
        Returns:
            Tuple of (trial_info, list of model responses)
        """
        self.logger.info(f"Starting trial {trial_number}")
        
        # Generate stimulus
        trial_info = self.generate_trial_stimulus(trial_number)
        
        # Query models
        try:
            responses = await self.vision_interface.run_trial(
                first_image_b64=trial_info['first_image_b64'],
                second_image_b64=trial_info['second_image_b64'],
                trial_info=trial_info,
                models=models
            )
            
            # Update staircase based on model performance
            # Use first valid response for staircase (could be made more sophisticated)
            if responses:
                for response in responses:
                    if response.perceptual_choice in [1, 2]:
                        correct = response.perceptual_choice == trial_info['correct_answer']
                        
                        # Only update staircase with the first valid response
                        # to avoid multiple updates per trial
                        if not hasattr(self, '_staircase_updated_this_trial'):
                            self.staircase.update(correct)
                            self._staircase_updated_this_trial = True
                        
                        # Add to analyzer (process all valid responses)
                        self.analyzer.add_response(response, trial_info['correct_answer'])
                        
                        self.logger.info(
                            f"Trial {trial_number} - Model: {response.model_name}, "
                            f"Choice: {response.perceptual_choice}, "
                            f"Correct: {trial_info['correct_answer']}, "
                            f"Confidence: {response.confidence}, "
                            f"Accuracy: {correct}"
                        )
                
                # Reset the staircase update flag for the next trial
                if hasattr(self, '_staircase_updated_this_trial'):
                    delattr(self, '_staircase_updated_this_trial')
            
            # Save trial data to session directory
            self.save_trial_data(trial_info, responses, session_dir)
            
            return trial_info, responses
            
        except Exception as e:
            self.logger.error(f"Trial {trial_number} failed: {e}")
            return trial_info, []
    
    def save_trial_data(self, trial_info: Dict, responses: List[ModelResponse], session_dir: Path):
        """
        Save trial data to files.
        
        Args:
            trial_info: Trial information
            responses: Model responses
            session_dir: Session directory to save files in
        """
        # Prepare data for saving
        trial_data = {
            'trial_info': trial_info,
            'responses': [],
            'staircase_state': self.staircase.get_performance_stats()
        }
        
        for response in responses:
            trial_data['responses'].append({
                'model_name': response.model_name,
                'perceptual_choice': response.perceptual_choice,
                'confidence': response.confidence,
                'response_time': response.response_time,
                'raw_response': response.raw_response,
                'correct': response.perceptual_choice == trial_info['correct_answer']
            })
        
        # Save individual trial in the session directory
        trial_file = session_dir / f"trial_{trial_info['trial_number']:03d}.json"
        with open(trial_file, 'w') as f:
            json.dump(trial_data, f, indent=2, default=str)
    
    async def run_session(self, models: List[str] = None, save_images: bool = False):
        """
        Run a complete experimental session.
        
        Args:
            models: List of models to test ["gpt-4o", "claude"]
            save_images: Whether to save stimulus images
        """
        if models is None:
            models = ["gpt-4o", "claude"]
        
        self.logger.info(f"Starting session with {self.n_trials_per_session} trials")
        self.logger.info(f"Testing models: {models}")
        
        # Create session directory
        session_dir = self.output_dir / f"session_{self.session_id}"
        session_dir.mkdir(exist_ok=True)
        
        if save_images:
            images_dir = session_dir / "stimulus_images"
            images_dir.mkdir(exist_ok=True)
        
        all_responses = []
        
        for trial_num in range(1, self.n_trials_per_session + 1):
            try:
                trial_info, responses = await self.run_single_trial(trial_num, session_dir, models)
                all_responses.extend(responses)
                
                # Optional: save stimulus images
                if save_images and responses:
                    self.save_stimulus_images(trial_info, images_dir)
                
                # Brief pause between trials
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Failed to complete trial {trial_num}: {e}")
                continue
        
        # Generate session summary
        self.generate_session_summary(session_dir, all_responses)
        
        self.logger.info(f"Session completed. Results saved to {session_dir}")
    
    def save_stimulus_images(self, trial_info: Dict, images_dir: Path):
        """Save stimulus images for debugging/visualization."""
        import base64
        from PIL import Image
        from io import BytesIO
        
        trial_num = trial_info['trial_number']
        
        # Save first interval image
        first_image_data = base64.b64decode(trial_info['first_image_b64'])
        first_image = Image.open(BytesIO(first_image_data))
        first_image_path = images_dir / f"trial_{trial_num:03d}_interval_1.png"
        first_image.save(first_image_path)
        
        # Save second interval image
        second_image_data = base64.b64decode(trial_info['second_image_b64'])
        second_image = Image.open(BytesIO(second_image_data))
        second_image_path = images_dir / f"trial_{trial_num:03d}_interval_2.png"
        second_image.save(second_image_path)
        
        self.logger.info(f"Saved stimulus images for trial {trial_num}")
        
        # Also save trial metadata with image info
        trial_metadata = {
            'trial_number': trial_num,
            'first_location': trial_info['first_location'],
            'second_location': trial_info['second_location'],
            'first_contrast': trial_info['first_contrast'],
            'second_contrast': trial_info['second_contrast'],
            'target_contrast': trial_info['target_contrast'],
            'correct_answer': trial_info['correct_answer'],
            'first_image_file': f"trial_{trial_num:03d}_interval_1.png",
            'second_image_file': f"trial_{trial_num:03d}_interval_2.png"
        }
        
        metadata_path = images_dir / f"trial_{trial_num:03d}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(trial_metadata, f, indent=2, default=str)
    
    def generate_session_summary(self, session_dir: Path, all_responses: List[ModelResponse]):
        """
        Generate comprehensive session summary.
        
        Args:
            session_dir: Directory to save summary
            all_responses: All model responses from session
        """
        # Generate analyzer report
        summary_report = self.analyzer.get_summary_report()
        
        # Save text report
        with open(session_dir / "summary_report.txt", 'w') as f:
            f.write(summary_report)
        
        # Save detailed metrics as JSON
        models = list(set(r.model_name for r in all_responses))
        detailed_metrics = {}
        
        for model in models:
            detailed_metrics[model] = self.analyzer.calculate_performance_metrics(model)
        
        # Add staircase information
        detailed_metrics['staircase'] = {
            'final_stats': self.staircase.get_performance_stats(),
            'threshold_estimate': self.staircase.get_threshold_estimate(),
            'converged': self.staircase.is_converged(),
            'n_reversals': len(self.staircase.reversals)
        }
        
        with open(session_dir / "detailed_metrics.json", 'w') as f:
            json.dump(detailed_metrics, f, indent=2, default=str)
        
        # Create CSV for further analysis
        self.create_analysis_csv(session_dir, all_responses)
        
        self.logger.info("Session summary generated")
        print(f"\n{summary_report}")
    
    def create_analysis_csv(self, session_dir: Path, all_responses: List[ModelResponse]):
        """Create CSV file for detailed statistical analysis."""
        data_rows = []
        
        for response in all_responses:
            # Get corresponding trial info
            trial_info = response.stimulus_info
            
            row = {
                'session_id': self.session_id,
                'trial_id': response.trial_id,
                'model': response.model_name,
                'perceptual_choice': response.perceptual_choice,
                'confidence': response.confidence,
                'response_time': response.response_time,
                'correct_answer': trial_info.get('correct_answer', -1),
                'correct': response.perceptual_choice == trial_info.get('correct_answer', -1),
                'target_contrast': trial_info.get('target_contrast', -1),
                'first_contrast': trial_info.get('first_contrast', -1),
                'second_contrast': trial_info.get('second_contrast', -1),
                'first_location': trial_info.get('first_location', -1),
                'second_location': trial_info.get('second_location', -1),
                'timestamp': trial_info.get('timestamp', '')
            }
            
            data_rows.append(row)
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            df.to_csv(session_dir / "analysis_data.csv", index=False)
    
    def load_api_keys_from_file(self, config_file: str = "api_keys.json"):
        """
        Load API keys from configuration file.
        
        Args:
            config_file: Path to JSON file containing API keys
        """
        try:
            with open(config_file, 'r') as f:
                keys = json.load(f)
            
            if 'openai_key' in keys and keys['openai_key']:
                self.vision_interface.openai_client = openai.OpenAI(api_key=keys['openai_key'])
                self.logger.info("OpenAI API key loaded")
            
            if 'anthropic_key' in keys and keys['anthropic_key']:
                self.vision_interface.anthropic_client = anthropic.Anthropic(api_key=keys['anthropic_key'])
                self.logger.info("Anthropic API key loaded")
            
            if 'gemini_key' in keys and keys['gemini_key']:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=keys['gemini_key'])
                    self.vision_interface.gemini_client = genai.GenerativeModel('gemini-1.5-pro')
                    self.logger.info("Gemini API key loaded")
                except ImportError:
                    self.logger.warning("google-generativeai not installed, skipping Gemini initialization")
                
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_file} not found")
        except Exception as e:
            self.logger.error(f"Failed to load API keys: {e}")

class ExperimentConfig:
    """Configuration class for experiment parameters."""
    
    DEFAULT_CONFIG = {
        "n_trials_per_session": 50,
        "target_performance": 0.71,
        "models_to_test": ["gpt-4o", "claude"],
        "staircase_params": {
            "initial_contrast": 0.5,
            "step_size_up": 0.05,
            "step_size_down": 0.02,
            "min_contrast": 0.1,
            "max_contrast": 1.0
        },
        "stimulus_params": {
            "image_size": 400,
            "patch_size": 60,
            "spatial_freq": 0.1
        }
    }
    
    @classmethod
    def load_from_file(cls, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return {**cls.DEFAULT_CONFIG, **config}
        except FileNotFoundError:
            return cls.DEFAULT_CONFIG
    
    @classmethod
    def save_default_config(cls, config_file: str = "experiment_config.json"):
        """Save default configuration to file."""
        with open(config_file, 'w') as f:
            json.dump(cls.DEFAULT_CONFIG, f, indent=2)

# Main execution functions
async def run_experiment_from_config(config_file: str = "experiment_config.json"):
    """
    Run experiment from configuration file.
    
    Args:
        config_file: Path to configuration file
    """
    # Load configuration
    config = ExperimentConfig.load_from_file(config_file)
    
    # Initialize experiment
    experiment = MetacognitionExperiment(
        n_trials_per_session=config["n_trials_per_session"],
        target_performance=config["target_performance"]
    )
    
    # Load API keys
    experiment.load_api_keys_from_file()
    
    # Run session
    await experiment.run_session(
        models=config["models_to_test"],
        save_images=config.get("save_images", False)
    )

def create_example_config():
    """Create example configuration and API key files."""
    # Create example config
    ExperimentConfig.save_default_config()
    
    # Create example API keys file
    example_keys = {
        "openai_key": "your-openai-api-key-here",
        "anthropic_key": "your-anthropic-api-key-here"
    }
    
    with open("api_keys_example.json", 'w') as f:
        json.dump(example_keys, f, indent=2)
    
    print("Created example configuration files:")
    print("- experiment_config.json (experiment parameters)")
    print("- api_keys_example.json (API key template)")
    print("\nPlease:")
    print("1. Copy api_keys_example.json to api_keys.json")
    print("2. Add your actual API keys to api_keys.json")
    print("3. Modify experiment_config.json as needed")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        create_example_config()
    else:
        # Run experiment
        asyncio.run(run_experiment_from_config()) 