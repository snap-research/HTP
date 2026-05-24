# HTP
This repository contains the code for [Hierarchical Token Prepending: Enhancing Information Flow in Decoder-based LLM Embeddings](https://arxiv.org/abs/2511.14868), accepted by **ACL 2026 (Oral)**.


# Model Directory

This directory contains zero-shot LLM-based embedding generation. 

## Overview

The model directory implements various large language model architectures optimized for embedding tasks, supporting different pooling strategies and parsing methods.

## Dependencies

### Basic LLM Block & Eval
- `transformers` version 4.55.2 
- `accelerate` - Multi-GPU and distributed training support
- `torch` - Core PyTorch library
- `datasets` - Hugging Face datasets library

### Torch Versions

- `pytorch-lightning` version 2.5.1.post0
- `torch` version 2.5.0+cu121
- `torchaudio` version 2.5.0+cu121
- `torchmetrics` version 1.0.3
- `torchrec` version 1.0.0+cu121
- `torchsnapshot` version 0.1.0
- `torchvision` version 0.20.0+cu121


#### NLP Processing
- `spacy` - Natural language processing library
- `en_core_web_sm` - English language model for spaCy

```bash
pip install torch transformers accelerate datasets numpy scikit-learn tqdm pyyaml beir mteb spacy
python -m spacy download en_core_web_sm
```

#### Evaluation Block
- `beir` - Benchmark for Information Retrieval evaluation

## How to run the code

### Optional: Downloading the weights to local

To disallow transformer autoupdates causing code inconsistency, we can download the code to the local directory by running:

```bash
python download_llm_weights.py
```

### MTEB Retrieval Tasks

Vanilla with mean embedding

```bash
python mteb_retrieval_tasks.py --model_method vanilla-mean-pool --architecture mistral-instruct-vanilla
```

Echo with mean embedding

```bash
python mteb_retrieval_tasks.py --model_method echo-mean-pool --architecture mistral-instruct-vanilla
```

Gloal TP with mean embedding

```bash
python mteb_retrieval_tasks.py --model_method tp-mean-pool --architecture mistral-instruct-tp
```

PromptEOL+ TP with last embedding

```bash
python mteb_retrieval_tasks.py --model_method tp-prompteol-last-pool --architecture mistral-instruct-tp
```

Sentence TP with mean embedding

```bash
python mteb_retrieval_tasks.py --model_method vanilla-mean-pool --architecture mistral-instruct-tp --use_which_plan tp_sentence
```

Hierachical TP with mean embedding

```bash
python mteb_retrieval_tasks.py --model_method vanilla-mean-pool --architecture mistral-instruct-tp --use_which_plan tp_sentence_begin
```

Results can be found in results/method_name/dataset.json files.
Note: If such files exist, the results will be automatically loaded from the json files instead of running the evaluation again.

#### Additional Parameters to consider

##### Model Configuration
- `--pooling` - Override pooling strategy (`mean`, `last`)
- `--output_layer` - Extract embeddings from specific layer (e.g., `-2` for second-to-last)
- `--model_name_or_path` - Use custom model path instead of config default

##### Token Positioning Parameters
- `--tp_starting_index` - Starting layer for token positioning modifications
- `--tp_exiting_index` - Ending layer for token positioning modifications
- `--global_sentence_tp` - Enable global sentence-level token positioning

##### System Configuration
- `--cuda_visible_devices` - Control GPU visibility (e.g., `"0,1,2,3"`)
- `--verbose` - Add will provide a debug on PST/EOS
- `--padding_side` - Sequence padding side (`left` or `right`)

##### Dataset Selection
- `--retrieval_datasets` - Specify datasets to evaluate (e.g., `--retrieval_datasets NFCorpus,FiQA2018`) It is suggested to change it in config.yaml

##### Example with Custom Parameters
```bash
python mteb_retrieval_tasks.py \
    --model_method tp-prompteol-last-pool \
    --architecture mistral-instruct-tp \
    --output_layer -1 \
    --tp_starting_index 10 \
    --tp_exiting_index 20 \
    --verbose 1 \
    --retrieval_datasets NFCorpus FiQA2018 SciFact
```

#### Change different datasets

In config.yaml, we can change the tasks in retrieval_task-tasks.

Suggested tasks: "SCIDOCS" #ClimateFEVER, #ArguAna #DBPedia, #FiQA2018 #ArguAna,FiQA2018,NFCorpus,SciFact,

### LongEmbed Long context retrieval tasks

Echo with mean embedding

```bash
python longembed_retrieval_tasks.py --model_method echo-mean-pool --architecture mistral-instruct-vanilla
```

Gloal TP with mean embedding

```bash
python longembed_retrieval_tasks.py --model_method tp-mean-pool --architecture mistral-instruct-tp
```

PromptEOL+ TP with last embedding

```bash
python longembed_retrieval_tasks.py --model_method tp-prompteol-last-pool --architecture mistral-instruct-tp
```

Sentence TP with mean embedding

```bash
python longembed_retrieval_tasks.py --model_method vanilla-mean-pool --architecture mistral-instruct-tp --use_which_plan tp_sentence
```

Hierachical TP with mean embedding

```bash
python longembed_retrieval_tasks.py --model_method vanilla-mean-pool --architecture mistral-instruct-tp --use_which_plan tp_sentence_begin
```

#### Datasets selection
tasks: "2wikimqa,summ_screen_fd,qmsum" 
`max_length` in config.yaml long_context_task limits the passage context lengths - may need to change it for echo embed to avoid OOM.


#### Synthetic Needdle, Passkey Tasks

Change the python bash files:

```bash
python synthetic_retrieval_tasks.py --model_method vanilla-mean-pool --architecture mistral-instruct-tp --use_which_plan tp_sentence_begin
```

`max_length` in config.yaml long_context_synthetic_task selects the corresponding needle/passkey contexts.

### STS tasks

Change the python bash files:

```bash
python mteb_sts_tasks.py --model_method vanilla-mean-pool --architecture mistral-instruct-tp --use_which_plan tp_sentence_begin
```

## Token Positioning (TP) Methods Description

### **Vanilla**
- **What it does**: Standard Mistral model behavior without any token positioning modifications
- **Implementation**: Uses the original Mistral architecture as-is for embedding generation
- **Use case**: Baseline comparison method that processes text normally through the transformer layers

### **TP (Token Positioning)**
- **What it does**: Modifies token positions during forward pass to improve embedding quality
- **Implementation**: 
  - Introduces special tokens (PST - Position Sensitive Tokens) at strategic locations
  - Modifies attention patterns between specified layers (`tp_starting_index` to `tp_exiting_index`)
  - Allows the model to better understand token relationships and context boundaries
- **Use case**: Global token positioning that affects the entire sequence processing

### **TP Sentence**
- **What it does**: Applies token positioning specifically at sentence boundaries
- **Implementation**:
  - Uses spaCy to detect sentence boundaries in the input text
  - Inserts PST tokens at the end of each sentence
  - Modifies attention mechanisms to respect sentence-level structure
- **Use case**: Sentence-aware embedding generation that preserves semantic boundaries

### **Echo**
- **What it does**: Repeats input text twice to enhance representation learning
- **Implementation**: Concatenates the input text with itself, allowing the model to process the same content multiple times
- **Use case**: Improves embedding quality through repetition and self-attention mechanisms

### **Key Configuration Parameters**
- `--use_which_plan vanilla`: No token positioning
- `--use_which_plan tp`: Global token positioning  
- `--use_which_plan tp_sentence`: Sentence-level token positioning
- `--use_which_plan tp_sentence_begin`: Hierarchical sentence positioning

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{ding2025hierarchical,
  title={Hierarchical Token Prepending: Enhancing Information Flow in Decoder-based LLM Embeddings},
  author={Ding, Xueying and Huang, Xingyue and Ju, Mingxuan and Collins, Liam and Liu, Yozen and Akoglu, Leman and Shah, Neil and Zhao, Tong},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026},
  url={https://arxiv.org/abs/2511.14868}
}
```
