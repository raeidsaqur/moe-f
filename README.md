# Filtered not Mixed: Filtering-Based Online Gating for Mixture of Large Language Models 

This repository serves as the supplementary materials reposity for our proposed Mixture of Experts Filter (MoE-F) work published and presented at the ICLR 2025 [[OpenReview](https://openreview.net/pdf?id=ecIvumCyAj)].


## 📋 Table of Contents

- [🧠 Concept](#concept)
- [📌 Concrete Example](#concrete-example)
  - [Financial Market Movement Task](#fmm-task)
  - [Expert Weights Heatmap](#expert-weights-heatmap)
- [📊 Generating the Results Tables](#generating-the-results-tables)
- [📈 Experts Performance on NIFTY Test Split](#experts-performance-on-nifty-test-split)



## 🧠 Concept
The following is a conceptual flow showing how MoE-F works: 
![Mixture of Experts](./imgs/mixture_of_experts_v3_8fps.gif)

## 📌 Concrete Example
### Financial Market Movement Task <a name="fmm-task"></a>

Examining a cross-sectional time-window snapshot allows a better understanding. 

![Market Movement Plot](./imgs/market_movement_plot.png)

### Expert Weights Heatmap <a name="expert-weights-heatmap"></a>
The below diagram depicts corresponding weighting ranks of the 7 experts corresponding to the 3 randomly sampled week-long trading windows with market mimicking different (bull, bear, neutral) regimes shown [above](#fmm-task). 

![Expert Weights Heatmap](./imgs/expert_weights_heatmap_coolwarm.png)


## 📊 Generating the Results Tables

The [experiments](./MoE-F_supplementary.materials/experiments) folder contains all expert models' results in response 
to the [NIFTY-LM](https://huggingface.co/datasets/raeidsaqur/NIFTY)'s _test split_.


To generate the main results of the paper (in Table 2), run:
```bash
./generate_results.sh --model_name "OpenAI" --model_variant "gpt-4o" --seed 42 --average "weighted"
```
Using the `model_names` and `model_variants` as desired.

```angular2html
model_names = ["Llama-2", "Llama-2", "Meta-Llama-3", "Meta-Llama-3", "Mixtral-8x7B", "dbrx", "OpenAI"]
model_variants = ["7b-chat-hf", "70b-chat-hf", "8B-Instruct", "70B-Instruct", "Instruct-v0.1", "instruct", "gpt-4o"]
```


### 📈 Experts Performance on NIFTY Test Split <a name="experts-performance-on-nifty-test-split"></a>

**Llama-class models (Llama 2,3-[7B, 8B, 70B])**

![Llama-class models' Confusion Matrices](imgs/llama_confusion_matrices_seed_42.png)

**MoE models (Mixtral_7x8B, DBRX) and GPT4o**

![MoEs and GPT4o models' Confusion Matrices](imgs/moe_gpt_confusion_matrices_seed_42.png)
