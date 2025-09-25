# LLM Cybersecurity

This repo contains the implementation for "Fine-tuning of Large Language Models for Domain-Specific Cybersecurity Knowledge".

## Setup
- Tested Python Version: 3.8.20 with Mamba
- Install dependencies
    ```
    pip install -e .[torch,metrics]
    ```


## Data Format
- Data format can be referenced from `data/data.json`

## Training
- LoRA Fine-tuning
  ```
  bash train_lora.sh
  ```
- QLoRA Fine-tuning
  ```
  bash train_qlora.sh
  ```
- Full Fine-tuning
  ```
  bash train_full.sh
  ```

## Download pretrained models for inference and evaluation
- Run `download_saves.py`
  ```
  python download_saves.py 
  ```

## Evaluation
- LoRA Fine-tuning
  ```
  bash eval.sh
  ```
- QLoRA Fine-tuning
  ```
  bash qlora_eval.sh
  ```
- Full Fine-tuning
  ```
  bash full_eval.sh
  ```
  
## Inference
- LoRA Fine-tuning
  ```
  bash infer_lora.sh 
  ```
- QLoRA Fine-tuning
  ```
  bash infer_qlora.sh 
  ```
- Full Fine-tuning
  ```
  bash infer_full.sh 
  ```

