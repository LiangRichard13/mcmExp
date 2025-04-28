

## **Comparative Experiments**

This experiment compares:

1. The model before and after fine-tuning
2. The fine-tuned model against existing large language models in the medical and traditional Chinese medicine (TCM) domains

###  Benchmark Details

- **Dataset:** CMMLU Traditional-Chinese-Medicine-Benchmark [(source)](https://huggingface.co/datasets/shuyuej/CMMLU-Traditional-Chinese-Medicine-Benchmark)
- **Location**: `./data/test_tcm_benchmark`
- **Size:**Contains 185 questions-answers
- **Evaluation method:** Rule-based scoring (model receives credit if its response contains key answer phrases)

###  Model Preparation

- **Base model**: `internlm2_5-7b-chat`

- **Fine-tuning method:** QLoRA

- **Training data:**
  - Source: [ShenNong_TCM_Dataset](https://huggingface.co/datasets/michaelwzhu/ShenNong_TCM_Dataset) 
  - Quality enhancement: Processed using `deepseek-v3`
  - Enhanced data location: `./data/train_improved_qa/improved_output_converted.json`(16,314 QA pairs)
  
- **Parameter scale:** All compared models have similar parameter counts

- **Training config file:**`./train/internlm2_5_chat_7b_qlora_tcm.py`


#### Training Process

![Train Process](./train/training_analysis.png)

- **Key training metrics**
  - Total training steps: 1243
  - Learning rate: Initial=0.00000484, Max=0.00020000, Final=0.00000000
  - Loss values: Initial=1.5594, Min=0.5230, Final=0.5573
  - Loss reduction: 1.0021 (64.26%)
  - Average training time per step: 1.3943s
  - Average data loading time: 0.0150s
  - Average memory usage: 15487.87 MB
- **Visualization data**: `./train/20250427_083447/vis_data`


### Experiment Result

|                Model Name                |  Accuracy  |
| :--------------------------------------: | :--------: |
| carebot_medical_multi_llama3-8b-instruct |   61.62%   |
|             huatuo_gpt_2-7b              |   61.08%   |
|             huatuo_gpt_o1-7b             |   83.24%   |
|           internlm2_5-7b-chat            |   78.38%   |
|            internlm2_5-7b-sft            | **86.49%** |

- The comparative experimental results demonstrate that the fine-tuned model (e.g., `internlm2_5-7b-sft`, 86.49%) achieves outstanding performance in Traditional Chinese Medicine (TCM) tasks. It significantly outperforms other TCM-specific models (e.g., `huatuo_gpt_2-7b`, 61.08%) and general medical models (e.g., `carebot_medical_multi_llama3-8b-instruct`, 61.62%), while matching the performance of the current state-of-the-art TCM reasoning model `huatuo_gpt-o1-7b` (83.24%). This conclusively validates the effectiveness of the fine-tuning approach for TCM knowledge processing tasks.
- **Interaction log:**

|           Model Name           |  Accuracy  | Improvement Over The Base Model |
| :----------------------------: | :--------: | :-----------------------------: |
| internlm2_5-7b-chat (baseline) |   78.38%   |                -                |
|             + RAG              |   80.54%   |             +2.16%              |
|          + Graph RAG           |   82.70%   |             +4.32%              |
|       + RAG + Graph RAG        | **85.41%** |             +7.03%              |

|          Model Name           |  Accuracy  | Improvement Over The Base Model |
| :---------------------------: | :--------: | :-----------------------------: |
| internlm2_5-7b-sft (baseline) |   86.49%   |                -                |
|             + RAG             |   84.32%   |             -2.17%              |
|          + Graph RAG          |   89.19%   |             +2.70%              |
|    + RAG + Graph RAG(MCM)     | **89.19%** |             +2.70%              |