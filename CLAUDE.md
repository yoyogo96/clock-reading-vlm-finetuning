# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Clock Reading Dataset Generator for fine-tuning LLMs to accurately read analog clocks. The project generates synthetic clock images with systematic reasoning processes for time reading.

## Development Commands

### Dataset Generation
```bash
# Install dependencies
pip install -r requirements.txt

# Generate a single test clock
python clock_generator.py

# Generate a small dataset (50 samples)
python example_usage.py

# Generate a full dataset (customizable)
python dataset_generator.py --num_samples 1000 --output_dir dataset

# Generate large dataset with custom settings
python dataset_generator.py --num_samples 5000 --image_size 512 --num_workers 8
```

### VLM Fine-tuning
```bash
# Quick start (recommended for beginners)
python quick_start.py --mode all --num_epochs 5

# Check environment and requirements
python quick_start.py --mode check

# Test data loading
python quick_start.py --mode test

# Train VLM model
python train_clock_vlm.py --batch_size 8 --num_epochs 10 --use_wandb

# Evaluate trained model
python inference.py --model_path checkpoints/best_model --mode evaluate

# Single image inference
python inference.py --model_path checkpoints/best_model --mode single --image_path example_clock.png --reasoning
```

### Data Testing
```bash
# Test dataset loader
python clock_dataset.py

# Test inference pipeline
python inference.py --model_path Salesforce/blip2-opt-2.7b --mode single --image_path example_clock.png
```

## Code Architecture

### Core Components

- `clock_generator.py`: Core clock image generation and reasoning logic
  - `ClockGenerator`: Main class for generating individual clock images
  - Handles multiple clock styles (classic, modern, vintage)
  - Generates 5-step reasoning processes for time reading

- `dataset_generator.py`: Large-scale dataset generation with parallel processing
  - `DatasetGenerator`: Manages batch generation and dataset organization
  - Supports train/val/test splits and multiple output formats
  - Includes statistical analysis and reporting

- `example_usage.py`: Demonstration scripts showing various use cases

### Key Features

1. **Clock Image Generation**: Creates diverse analog clock images with different styles
2. **Systematic Reasoning**: 5-step process for teaching time reading logic
3. **Parallel Processing**: Efficient generation using ThreadPoolExecutor
4. **Multiple Formats**: Outputs JSON, JSONL for different ML frameworks
5. **Difficulty Levels**: Easy (quarter hours) to hard (irregular minutes)

### Reasoning Process Structure

Each generated sample includes a 5-step reasoning process:
1. Clock structure identification
2. Hand identification (hour vs minute)  
3. Hour hand position analysis
4. Minute hand position analysis
5. Final time conclusion

This systematic approach helps LLMs learn the logical process of reading analog clocks rather than just pattern matching.