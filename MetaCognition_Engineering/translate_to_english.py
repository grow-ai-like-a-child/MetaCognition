#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translate Chinese comments and strings in Python files to English
"""

import os
import re
from pathlib import Path

# Translation dictionary
translations = {
    # Function docstrings
    "Load correct answers dictionary": "Load correct answers dictionary",
    "Load model results": "Load model results",
    "Analyze model results": "Analyze model results",
    "Print analysis results": "Print analysis results",
    "Create confidence distribution plots": "Create confidence distribution plots",
    "Create detailed question-by-question comparison analysis": "Create detailed question-by-question comparison analysis",
    "Generate summary statistics": "Generate summary statistics",
    
    # Comments
    "Group by original_qid, merge stage1 and stage2": "Group by original_qid, merge stage1 and stage2",
    "Only analyze complete results with both stage1 and stage2": "Only analyze complete results with both stage1 and stage2",
    "Get correct answer": "Get correct answer",
    "Merge stage1 and stage2 information": "Merge stage1 and stage2 information",
    "Group by task type": "Group by task type",
    "Calculate statistics for each task type": "Calculate statistics for each task type",
    "Confidence distribution": "Confidence distribution",
    "Average confidence": "Average confidence",
    "Average latency": "Average latency",
    "Overall statistics": "Overall statistics",
    "Set Chinese font": "Set Chinese font",
    "Task accuracy comparison": "Task accuracy comparison",
    "Display percentages on bars": "Display percentages on bars",
    "Confidence distribution": "Confidence distribution",
    "Add numerical annotations": "Add numerical annotations",
    "Loading data": "Loading data",
    "Analyzing results": "Analyzing results",
    "Printing results": "Printing results",
    "Creating charts": "Creating charts",
    "Skipping chart generation": "Skipping chart generation",
    "Save detailed results to CSV": "Save detailed results to CSV",
    
    # Print statements
    "Total of": "Total of",
    "complete results (including stage1 and stage2)": "complete results (including stage1 and stage2)",
    "task": "task",
    "Total questions": "Total questions",
    "Correct answers": "Correct answers",
    "Accuracy": "Accuracy",
    "Average confidence": "Average confidence",
    "Average latency": "Average latency",
    "Confidence distribution": "Confidence distribution",
    "Overall statistics": "Overall statistics",
    "总Correct answers": "Total correct",
    "总体Accuracy": "Overall accuracy",
    "正在Loading data": "Loading data",
    "Loaded": "Loaded",
    "correct answers": "correct answers",
    "model results": "model results",
    "正在Analyzing results": "Analyzing results",
    "Creating charts时出错": "Error creating charts",
    "Skipping chart generation": "Skipping chart generation",
    "Detailed results saved to": "Detailed results saved to",
    "Chart saved to": "Chart saved to",
    
    # Argument descriptions
    "Qwen2-VL metacognitive reasoning accuracy analysis": "Qwen2-VL metacognitive reasoning accuracy analysis",
    "Correct answers file path": "Correct answers file path",
    "Model results file path": "Model results file path",
    "Output CSV file path": "Output CSV file path",
    "Output chart file path": "Output chart file path",
    "Qwen2-VL metacognitive reasoning comparison analysis": "Qwen2-VL metacognitive reasoning comparison analysis",
    "Output JSON file path": "Output JSON file path",
}

def translate_file(file_path):
    """Translate a single Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply translations
        for chinese, english in translations.items():
            content = content.replace(chinese, english)
        
        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Translated: {file_path}")
            return True
        else:
            print(f"⏭️  No changes: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error translating {file_path}: {e}")
        return False

def main():
    """Main function"""
    print("🌐 Translating Python files to English...")
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    translated_count = 0
    total_count = len(python_files)
    
    for file_path in python_files:
        if translate_file(file_path):
            translated_count += 1
    
    print(f"\n📊 Translation Summary:")
    print(f"  Total files: {total_count}")
    print(f"  Translated: {translated_count}")
    print(f"  No changes: {total_count - translated_count}")

if __name__ == "__main__":
    main()
