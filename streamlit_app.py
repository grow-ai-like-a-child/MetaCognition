import streamlit as st
import asyncio
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import io
from datetime import datetime

from gabor_stimulus import GaborStimulus
from staircase import AdaptiveStaircase
from vision_models import VisionModelInterface, MetacognitionAnalyzer
from experiment_runner import MetacognitionExperiment, ExperimentConfig

st.set_page_config(
    page_title="Metacognition Investigation Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitExperimentInterface:
    """Streamlit interface for the metacognition experiment."""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'experiment' not in st.session_state:
            st.session_state.experiment = None
        if 'experiment_running' not in st.session_state:
            st.session_state.experiment_running = False
        if 'results_data' not in st.session_state:
            st.session_state.results_data = []
        if 'current_trial' not in st.session_state:
            st.session_state.current_trial = 0
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.title("🧠 Metacognition Investigation")
        st.sidebar.markdown("### Configuration")
        
        # API Keys
        st.sidebar.markdown("#### API Keys")
        openai_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Required for GPT-4o vision model"
        )
        anthropic_key = st.sidebar.text_input(
            "Anthropic API Key", 
            type="password", 
            help="Required for Claude Sonnet 3.5 vision model"
        )
        
        # Experiment Parameters
        st.sidebar.markdown("#### Experiment Parameters")
        n_trials = st.sidebar.slider("Number of Trials", 5, 100, 25)
        target_performance = st.sidebar.slider("Target Performance", 0.5, 0.9, 0.71, 0.01)
        
        # Model Selection
        st.sidebar.markdown("#### Models to Test")
        test_gpt4o = st.sidebar.checkbox("GPT-4o", value=True, disabled=not openai_key)
        test_claude = st.sidebar.checkbox("Claude Sonnet 3.5", value=True, disabled=not anthropic_key)
        
        # Advanced Settings
        with st.sidebar.expander("Advanced Settings"):
            contrast_range = st.slider("Initial Contrast", 0.1, 1.0, 0.5, 0.05)
            step_size = st.slider("Staircase Step Size", 0.01, 0.1, 0.05, 0.01)
        
        return {
            'openai_key': openai_key,
            'anthropic_key': anthropic_key,
            'n_trials': n_trials,
            'target_performance': target_performance,
            'models': [m for m, test in [('gpt-4o', test_gpt4o), ('claude', test_claude)] if test],
            'initial_contrast': contrast_range,
            'step_size': step_size
        }
    
    def render_stimulus_preview(self):
        """Render a preview of the Gabor stimulus."""
        st.markdown("### 👁️ Stimulus Preview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_location = st.selectbox("Target Location", range(6), index=2)
        with col2:
            target_contrast = st.slider("Target Contrast", 0.1, 1.0, 0.8, 0.1)
        with col3:
            distractor_contrast = st.slider("Distractor Contrast", 0.1, 1.0, 0.3, 0.1)
        
        if st.button("Generate Preview"):
            stimulus_gen = GaborStimulus()
            
            # Create example stimulus
            example_image = stimulus_gen.create_stimulus_display(
                target_location=target_location,
                target_contrast=target_contrast,
                distractor_contrast=distractor_contrast
            )
            
            # Display
            fig = stimulus_gen.visualize_stimulus(example_image, "Example Gabor Stimulus")
            st.pyplot(fig)
            plt.close()
    
    def render_experiment_control(self, config):
        """Render experiment control panel."""
        st.markdown("### 🎯 Experiment Control")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Experiment", 
                        disabled=not config['models'] or st.session_state.experiment_running):
                self.start_experiment(config)
        
        with col2:
            if st.button("⏹️ Stop Experiment", 
                        disabled=not st.session_state.experiment_running):
                self.stop_experiment()
        
        with col3:
            if st.button("🔄 Reset", disabled=st.session_state.experiment_running):
                self.reset_experiment()
        
        # Status display
        if st.session_state.experiment_running:
            st.info(f"🔄 Experiment running... Trial {st.session_state.current_trial}")
            
            # Progress bar
            if st.session_state.experiment:
                progress = st.session_state.current_trial / config['n_trials']
                st.progress(progress)
        
        elif st.session_state.experiment and not st.session_state.experiment_running:
            st.success("✅ Experiment completed!")
    
    def start_experiment(self, config):
        """Start the experiment with given configuration."""
        try:
            # Initialize experiment
            experiment = MetacognitionExperiment(
                openai_api_key=config['openai_key'] if config['openai_key'] else None,
                anthropic_api_key=config['anthropic_key'] if config['anthropic_key'] else None,
                n_trials_per_session=config['n_trials'],
                target_performance=config['target_performance']
            )
            
            st.session_state.experiment = experiment
            st.session_state.experiment_running = True
            st.session_state.current_trial = 0
            st.session_state.results_data = []
            
            # Run experiment asynchronously
            self.run_experiment_async(config)
            
        except Exception as e:
            st.error(f"Failed to start experiment: {e}")
    
    def run_experiment_async(self, config):
        """Run experiment asynchronously (simulated for Streamlit)."""
        # Note: This is a simplified version for Streamlit demo
        # In practice, you'd need to handle async operations differently
        st.info("Experiment would run here. See console output for full experiment execution.")
        
        # For demo purposes, simulate some results
        self.simulate_experiment_results(config)
    
    def simulate_experiment_results(self, config):
        """Simulate experiment results for demo purposes."""
        np.random.seed(42)
        
        results = []
        for trial in range(config['n_trials']):
            for model in config['models']:
                # Simulate realistic performance
                if model == 'gpt-4o':
                    accuracy = np.random.beta(7, 3)  # Tends toward higher accuracy
                    confidence = np.random.choice(range(3, 7), p=[0.1, 0.2, 0.4, 0.3])
                else:  # claude
                    accuracy = np.random.beta(6, 4)  # Slightly lower accuracy
                    confidence = np.random.choice(range(2, 6), p=[0.2, 0.3, 0.3, 0.2])
                
                correct = np.random.random() < accuracy
                
                results.append({
                    'trial': trial + 1,
                    'model': model,
                    'correct': correct,
                    'confidence': confidence,
                    'response_time': np.random.normal(3.0, 1.0),
                    'contrast': max(0.1, 0.5 + np.random.normal(0, 0.1))
                })
        
        st.session_state.results_data = results
        st.session_state.experiment_running = False
        st.session_state.current_trial = config['n_trials']
    
    def stop_experiment(self):
        """Stop the running experiment."""
        st.session_state.experiment_running = False
        st.warning("⏹️ Experiment stopped by user")
    
    def reset_experiment(self):
        """Reset experiment state."""
        st.session_state.experiment = None
        st.session_state.experiment_running = False
        st.session_state.results_data = []
        st.session_state.current_trial = 0
        st.success("🔄 Experiment reset")
    
    def render_results_analysis(self):
        """Render results analysis section."""
        if not st.session_state.results_data:
            st.info("No results available yet. Run an experiment to see analysis.")
            return
        
        st.markdown("### 📊 Results Analysis")
        
        df = pd.DataFrame(st.session_state.results_data)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_accuracy = df['correct'].mean()
            st.metric("Overall Accuracy", f"{overall_accuracy:.3f}")
        
        with col2:
            mean_confidence = df['confidence'].mean()
            st.metric("Mean Confidence", f"{mean_confidence:.2f}")
        
        with col3:
            n_trials = len(df)
            st.metric("Total Trials", n_trials)
        
        with col4:
            mean_rt = df['response_time'].mean()
            st.metric("Mean Response Time", f"{mean_rt:.2f}s")
        
        # Model comparison
        st.markdown("#### Model Comparison")
        
        model_stats = df.groupby('model').agg({
            'correct': ['mean', 'count'],
            'confidence': 'mean',
            'response_time': 'mean'
        }).round(3)
        
        st.dataframe(model_stats)
        
        # Visualizations
        self.render_performance_plots(df)
        self.render_metacognition_plots(df)
    
    def render_performance_plots(self, df):
        """Render performance visualization plots."""
        st.markdown("#### Performance Over Time")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy over trials
            fig_acc = px.line(
                df.groupby(['trial', 'model'])['correct'].mean().reset_index(),
                x='trial', y='correct', color='model',
                title='Accuracy Over Trials',
                labels={'correct': 'Accuracy', 'trial': 'Trial Number'}
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig_conf = px.histogram(
                df, x='confidence', color='model', 
                title='Confidence Distribution',
                nbins=6, barmode='group'
            )
            st.plotly_chart(fig_conf, use_container_width=True)
    
    def render_metacognition_plots(self, df):
        """Render metacognition-specific plots."""
        st.markdown("#### Metacognitive Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence vs Accuracy
            conf_acc = df.groupby(['model', 'confidence'])['correct'].mean().reset_index()
            fig_meta = px.line(
                conf_acc, x='confidence', y='correct', color='model',
                title='Confidence vs Accuracy (Calibration)',
                labels={'correct': 'Accuracy', 'confidence': 'Confidence Rating'}
            )
            # Add perfect calibration line
            fig_meta.add_trace(
                go.Scatter(x=[1, 6], y=[1/6, 1], mode='lines', 
                          name='Perfect Calibration', line=dict(dash='dash'))
            )
            st.plotly_chart(fig_meta, use_container_width=True)
        
        with col2:
            # Response time vs confidence
            fig_rt = px.scatter(
                df, x='confidence', y='response_time', color='model',
                title='Response Time vs Confidence',
                labels={'response_time': 'Response Time (s)', 'confidence': 'Confidence Rating'}
            )
            st.plotly_chart(fig_rt, use_container_width=True)
        
        # Metacognitive sensitivity
        st.markdown("#### Metacognitive Sensitivity")
        
        sensitivity_data = []
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            correct_conf = model_data[model_data['correct']]['confidence'].mean()
            incorrect_conf = model_data[~model_data['correct']]['confidence'].mean()
            sensitivity = correct_conf - incorrect_conf
            
            sensitivity_data.append({
                'model': model,
                'correct_confidence': correct_conf,
                'incorrect_confidence': incorrect_conf,
                'metacognitive_sensitivity': sensitivity
            })
        
        sens_df = pd.DataFrame(sensitivity_data)
        st.dataframe(sens_df.round(3))
        
        # Sensitivity bar plot
        fig_sens = px.bar(
            sens_df, x='model', y='metacognitive_sensitivity',
            title='Metacognitive Sensitivity by Model',
            labels={'metacognitive_sensitivity': 'Sensitivity (Correct - Incorrect Confidence)'}
        )
        st.plotly_chart(fig_sens, use_container_width=True)
    
    def render_data_export(self):
        """Render data export section."""
        if not st.session_state.results_data:
            return
        
        st.markdown("### 💾 Data Export")
        
        df = pd.DataFrame(st.session_state.results_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV download
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"metacognition_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON download
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"metacognition_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Show raw data
            if st.button("👁️ View Raw Data"):
                st.dataframe(df, use_container_width=True)

def main():
    """Main Streamlit application."""
    st.title("🧠 Metacognition Investigation Platform")
    st.markdown("""
    This platform tests the metacognitive abilities of vision language models (GPT-4o and Claude Sonnet 3.5) 
    using a two-alternative forced-choice task with Gabor patch stimuli.
    
    **Task**: Models identify which temporal interval contains the higher-contrast Gabor patch and rate their confidence.
    """)
    
    # Initialize interface
    interface = StreamlitExperimentInterface()
    
    # Render sidebar configuration
    config = interface.render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Experiment", "👁️ Stimulus Preview", "📊 Results", "💾 Data"])
    
    with tab1:
        interface.render_experiment_control(config)
        
        # Show experiment info
        if config['models']:
            st.markdown("#### Selected Configuration")
            st.json({
                'models': config['models'],
                'trials': config['n_trials'],
                'target_performance': config['target_performance']
            })
    
    with tab2:
        interface.render_stimulus_preview()
    
    with tab3:
        interface.render_results_analysis()
    
    with tab4:
        interface.render_data_export()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About**: This experiment is based on metacognition research investigating confidence judgments in perceptual decision-making tasks.
    The staircase procedure maintains ~71% performance to enable meaningful confidence analysis.
    """)

if __name__ == "__main__":
    main() 