#!/bin/bash

# Script to generate model evaluation results
#
# This script runs the filtering_moe_eval.py script with specified models, variants, and seeds.
#
# Usage:
# ./generate_results.sh --model_name "OpenAI" --model_variant "gpt-4o" --seed 42 --average "weighted"

function show_help() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --model_name       Model name (e.g., Llama-2, Meta-Llama-3, Mixtral-8x7B, dbrx, OpenAI)"
  echo "  --model_variant    Model variant (e.g., 7b-chat-hf, 70b-chat-hf, 8B-Instruct, gpt-4o)"
  echo "  --seed             Random seed (e.g., 0, 13, 42)"
  echo "  --average          Average method for evaluation (default: weighted)"
  echo "  --help             Show this help message and exit"
}

# Default values
average="weighted"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model_name)
      model_name="$2"
      shift
      shift
      ;;
    --model_variant)
      model_variant="$2"
      shift
      shift
      ;;
    --seed)
      seed="$2"
      shift
      shift
      ;;
    --average)
      average="$2"
      shift
      shift
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [[ -z "$model_name" || -z "$model_variant" || -z "$seed" ]]; then
  echo "Error: Missing required arguments."
  show_help
  exit 1
fi

# Run the filtering_moe_eval.py script with the provided arguments
python filtering_moe_eval.py --model_name "$model_name" --model_variant "$model_variant" --seed "$seed" --average "$average"

# Example Usage:
# ./generate_results.sh --model_name "OpenAI" --model_variant "gpt-4o" --seed 42 --average "weighted"

# Available Models and Variants:
# model_names = ["Llama-2", "Llama-2", "Meta-Llama-3", "Meta-Llama-3", "Mixtral-8x7B", "dbrx", "OpenAI"]
# model_variants = ["7b-chat-hf", "70b-chat-hf", "8B-Instruct", "70B-Instruct", "Instruct-v0.1", "instruct", "gpt-4o"]
# seeds = [0, 13, 42]
