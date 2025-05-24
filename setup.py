#!/usr/bin/env python3
"""
Setup script for the Metacognition Investigation Platform.
Initializes configuration files and tests basic functionality.
"""

import os
import json
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    directories = [
        "experiment_results",
        "experiment_results/logs",
        "experiment_results/data",
        "experiment_results/visualizations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_config_files():
    """Create configuration files if they don't exist."""
    
    # Create experiment configuration
    config_file = "experiment_config.json"
    if not os.path.exists(config_file):
        config = {
            "n_trials_per_session": 25,
            "target_performance": 0.71,
            "models_to_test": ["gpt-4o", "claude"],
            "save_images": False,
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
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Created configuration file: {config_file}")
    else:
        print(f"✓ Configuration file already exists: {config_file}")
    
    # Create API keys template
    api_keys_example = "api_keys_example.json"
    if not os.path.exists(api_keys_example):
        keys_template = {
            "openai_key": "your-openai-api-key-here",
            "anthropic_key": "your-anthropic-api-key-here"
        }
        
        with open(api_keys_example, 'w') as f:
            json.dump(keys_template, f, indent=2)
        print(f"✓ Created API keys template: {api_keys_example}")
    else:
        print(f"✓ API keys template already exists: {api_keys_example}")

def test_imports():
    """Test if all required modules can be imported."""
    print("\n🧪 Testing module imports...")
    
    required_modules = [
        "numpy",
        "matplotlib",
        "scipy",
        "cv2",
        "PIL",
        "pandas",
        "openai",
        "anthropic"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - NOT FOUND")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required modules are available!")
        return True

def test_local_modules():
    """Test if local modules can be imported."""
    print("\n🔧 Testing local modules...")
    
    local_modules = [
        "gabor_stimulus",
        "staircase", 
        "vision_models",
        "experiment_runner"
    ]
    
    success = True
    
    for module in local_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - ERROR: {e}")
            success = False
    
    return success

def test_stimulus_generation():
    """Test basic stimulus generation."""
    print("\n🎯 Testing stimulus generation...")
    
    try:
        from gabor_stimulus import GaborStimulus
        
        stimulus = GaborStimulus()
        test_image = stimulus.create_stimulus_display(
            target_location=2,
            target_contrast=0.8,
            distractor_contrast=0.3
        )
        
        print(f"✓ Generated stimulus image: {test_image.shape}")
        
        # Test base64 conversion
        b64_data = stimulus.image_to_base64(test_image)
        print(f"✓ Base64 conversion successful: {len(b64_data)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Stimulus generation failed: {e}")
        return False

def test_staircase():
    """Test staircase procedure."""
    print("\n📊 Testing staircase procedure...")
    
    try:
        from staircase import AdaptiveStaircase, simulate_staircase
        
        # Quick simulation
        staircase, results = simulate_staircase(n_trials=10)
        
        print(f"✓ Staircase simulation completed: {len(results)} trials")
        print(f"✓ Final performance: {staircase.get_performance_stats()['recent_performance']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Staircase test failed: {e}")
        return False

def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 Checking API key configuration...")
    
    api_keys_file = "api_keys.json"
    
    if not os.path.exists(api_keys_file):
        print(f"⚠️  API keys file not found: {api_keys_file}")
        print("Please copy api_keys_example.json to api_keys.json and add your API keys")
        return False
    
    try:
        with open(api_keys_file, 'r') as f:
            keys = json.load(f)
        
        openai_configured = keys.get('openai_key', '').startswith('sk-')
        anthropic_configured = keys.get('anthropic_key', '').startswith('sk-')
        
        if openai_configured:
            print("✓ OpenAI API key appears to be configured")
        else:
            print("⚠️  OpenAI API key not configured or invalid format")
        
        if anthropic_configured:
            print("✓ Anthropic API key appears to be configured")
        else:
            print("⚠️  Anthropic API key not configured or invalid format")
        
        return openai_configured or anthropic_configured
        
    except Exception as e:
        print(f"✗ Error reading API keys: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("🧠 METACOGNITION INVESTIGATION PLATFORM - SETUP COMPLETE")
    print("="*60)
    
    print("\n📚 NEXT STEPS:")
    print("1. Configure API keys:")
    print("   cp api_keys_example.json api_keys.json")
    print("   # Edit api_keys.json with your actual API keys")
    
    print("\n2. Run the web interface:")
    print("   streamlit run streamlit_app.py")
    
    print("\n3. Or run command-line experiment:")
    print("   python experiment_runner.py")
    
    print("\n4. Test individual components:")
    print("   python gabor_stimulus.py")
    print("   python staircase.py")
    print("   python vision_models.py")
    
    print("\n🔧 CONFIGURATION:")
    print("- Edit experiment_config.json to customize parameters")
    print("- Check logs in experiment_results/logs/")
    print("- Results saved to experiment_results/session_*/")
    
    print("\n📖 For more information, see README.md")

def main():
    """Main setup function."""
    print("🧠 Metacognition Investigation Platform Setup")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create configuration files
    print("\n⚙️  Creating configuration files...")
    create_config_files()
    
    # Test imports
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ Setup incomplete due to missing dependencies.")
        sys.exit(1)
    
    # Test local modules
    local_ok = test_local_modules()
    if not local_ok:
        print("\n❌ Setup incomplete due to local module errors.")
        sys.exit(1)
    
    # Test functionality
    stimulus_ok = test_stimulus_generation()
    staircase_ok = test_staircase()
    
    if not (stimulus_ok and staircase_ok):
        print("\n⚠️  Some functionality tests failed, but basic setup is complete.")
    
    # Check API keys
    check_api_keys()
    
    # Print instructions
    print_usage_instructions()
    
    print("\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main() 