import argparse
import os
import sys
import textwrap
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch
import tqdm
import yaml
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import mteb
from evaluator import compute_cosine_similarities
from model import Qwen2ForCausalLM, GemmaForCausalLM, MistralForCausalLM
from parser import Parser, create_add_pst_function
from pooling.pooling import Pooling
torch.set_num_threads(24)

class Config:
    def __init__(self, data=None, models=None, architectures=None):
        self.data = SimpleNamespace(**(data or {}))
        self.models = SimpleNamespace(**(models or {}))
        self.architectures = SimpleNamespace(**(architectures or {}))

def load_config_from_yaml(config_file="config.yaml"):
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"warning: config file {config_file} not found, using command line parameters")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"error: config file {config_file} format error: {e}")
        sys.exit(1)
    return yaml_config


def make_sentence_collate_functions(parser,
                                    sentence_process_function=None):
    def collate_fn(batch):
        grouped_batch = defaultdict(list)
        for d in batch:
            for k, v in d.items():
                grouped_batch[k].append(v)
        batch = dict(grouped_batch)
        if sentence_process_function is None:
            sentence1_tagged = [('document', {'text': q}) for q in batch['sentence1']]
            sentence2_tagged = [('document', {'text': q}) for q in batch['sentence2']]
        else:
            sentence1_tagged = [('document', {'text': sentence_process_function(q)}) for q in batch['sentence1']]
            sentence2_tagged = [('document', {'text': sentence_process_function(q)}) for q in batch['sentence2']]
        score = [s for s in batch['score']]
        return { "s1": parser(sentence1_tagged),
                "s2": parser(sentence2_tagged),
                "score": score,
                }
    return collate_fn




class EmbeddingWrapper(nn.Module):
    def __init__(self, 
                 model, 
                 tokenizer,
                 pooling,
                 parser,
                 name=None,
                 sentence_process_function=None,
                 batch_size=32,
                 output_layer = -2,
                 verbose = False):
        super(EmbeddingWrapper, self).__init__()
        self.model = model
        
        # Print model memory usage
        if hasattr(model, 'get_memory_footprint'):
            memory_mb = model.get_memory_footprint() / (1024 * 1024)
            print(f"Model memory usage: {memory_mb:.2f} MB")
        else:
            # Alternative method using model parameters
            total_params = sum(p.numel() * p.element_size() for p in model.parameters())
            memory_mb = total_params / (1024 * 1024)
            print(f"Model memory usage (estimated): {memory_mb:.2f} MB")
        self.name = name
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.output_layer = output_layer
        self.parser = parser
        self.verbose = verbose
        self.sentence_process_function = sentence_process_function
        self.batch_size = batch_size


    def encode(self, ds):
        collate_fn = make_sentence_collate_functions(self.parser, self.sentence_process_function)
        dataloader = DataLoader(ds, batch_size=self.batch_size, collate_fn=collate_fn)
        accelerator = Accelerator()
        dataloader = accelerator.prepare(dataloader)
        s1_embeddings = []
        s2_embeddings = []
        scores = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc='encoding', mininterval=10):
                s1_embed_mask = batch['s1']['embed_mask']
                s1_input_ids = batch['s1']['input_ids']
                s1_pst_positions = batch['s1']['pst_positions']
                s1_eos_positions = batch['s1']['eos_positions']
                s1_begin_pst_positions = batch['s1']['begin_pst_positions']

                s2_embed_mask = batch['s2']['embed_mask']
                s2_input_ids = batch['s2']['input_ids']
                s2_pst_positions = batch['s2']['pst_positions']
                s2_eos_positions = batch['s2']['eos_positions']
                s2_begin_pst_positions = batch['s2']['begin_pst_positions']

                if self.verbose:
                    for idx, (pst_pos, eos_pos) in enumerate(zip(s1_pst_positions, s1_eos_positions)):
                        print(f"Example S1 #{idx}")
                        print("PST positions:", pst_pos)
                        print("EOS positions:", eos_pos)
                        print("Begin PST positions:", s1_begin_pst_positions[idx])

                        input_ids = batch['s1']["input_ids"][idx]
                        decoded_tokens = self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
                        decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                        print("Decoded text:", decoded_text)
                        print("\n[PST] tokens at positions:")
                        for pos in pst_pos:
                            if pos < len(decoded_tokens):
                                print(f"  idx {pos}: {decoded_tokens[pos]}")
                        print("\n[EOS] tokens at positions:")
                        for pos in eos_pos:
                            if pos < len(decoded_tokens):
                                print(f"  idx {pos}: {decoded_tokens[pos]}")
                        print("\n[Begin PST] tokens at positions:")
                        for pos in s1_begin_pst_positions[idx]:
                            if pos < len(decoded_tokens):
                                print(f"  idx {pos}: {decoded_tokens[pos]}")
                        print("-" * 80)
                    
                    for idx, (pst_pos, eos_pos) in enumerate(zip(s2_pst_positions, s2_eos_positions)):
                        print(f"Example S2 #{idx}")
                        print("PST positions:", pst_pos)
                        print("EOS positions:", eos_pos)
                        print("Begin PST positions:", s2_begin_pst_positions[idx])

                        input_ids = batch['s2']["input_ids"][idx]
                        decoded_tokens = self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
                        decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                        print("Decoded text:", decoded_text)
                        print("\n[PST] tokens at positions:")
                        for pos in pst_pos:
                            if pos < len(decoded_tokens):
                                print(f"  idx {pos}: {decoded_tokens[pos]}")
                        print("\n[EOS] tokens at positions:")
                        for pos in eos_pos:
                            if pos < len(decoded_tokens):
                                print(f"  idx {pos}: {decoded_tokens[pos]}")
                        print("\n[Begin PST] tokens at positions:")
                        for pos in s2_begin_pst_positions[idx]:
                            if pos < len(decoded_tokens):
                                print(f"  idx {pos}: {decoded_tokens[pos]}")
                        print("-" * 80)

                raw_outputs_s1 = self.model(**batch['s1'],output_layer = self.output_layer)
                outputs_s1 = raw_outputs_s1.hidden_states
                outputs_s1 = self.pooling(token_embeddings = outputs_s1, embed_mask = s1_embed_mask)
                s1_embeddings.append(outputs_s1)

                raw_outputs_s2 = self.model(**batch['s2'],output_layer = self.output_layer)
                outputs_s2 = raw_outputs_s2.hidden_states
                outputs_s2 = self.pooling(token_embeddings = outputs_s2, embed_mask = s2_embed_mask)
                s2_embeddings.append(outputs_s2)

                scores.append(torch.tensor(batch['score'],dtype=torch.int))

            s1_embedding_result = torch.cat(s1_embeddings, dim=0)
            s2_embedding_result = torch.cat(s2_embeddings, dim=0)
            score_result = torch.cat(scores, dim=0)
            return s1_embedding_result, s2_embedding_result, score_result



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_method", type=str, default=None,
                        help="Select which method from models to use")
    parser.add_argument("--architecture", type=str, default=None,
                        help="Select which architecture to use")
    parser.add_argument("--pooling", type=str, default=None,
                        help="Change pooling method in models")
    parser.add_argument("--padding_side", type=str, default=None,
                        help="Change padding side in models")
    parser.add_argument("--global_sentence_tp", type=bool, default=None,
                        help="Change global_sentence_tp in models")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Change model_name_or_path in architectures")
    parser.add_argument("--use_which_plan", type=str, default=None,
                        help="Change use_which_plan in architectures")
    parser.add_argument("--output_layer", type=int, default=None,
                        help="Change output_layer in architectures")
    parser.add_argument("--verbose", action="store_true",
                        help="verbose mode for debugging")
    parser.add_argument("--tp_starting_index", type=int, default=None,
                        help="Change tp_starting_index in architectures")
    parser.add_argument("--tp_exiting_index", type=int, default=None,
                        help="Change tp_exiting_index in architectures")
    parser.add_argument("--cuda_visible_devices", type=str, default=None,
                        help="Change visible devices in gpu_config")
    parser.add_argument("--retrieval_datasets", type=str, nargs='+', default=None,
                        help="Pass a list of datasets for retrieval_task")
    args = parser.parse_args()
    yaml_config = load_config_from_yaml()

    # Load data task
    data = yaml_config['data']['sts_task']

    # Override retrieval datasets if provided
    if args.retrieval_datasets:
        data['tasks'] = ','.join(args.retrieval_datasets)

    # Load model method
    model_method = args.model_method or yaml_config.get('default_model_config')
    if model_method not in yaml_config['models']:
        print(f"error: model method '{model_method}' not found")
        print(f"available model methods: {list(yaml_config['models'].keys())}")
        sys.exit(1)
    models = yaml_config['models'][model_method]

    # Apply model-specific parser arguments
    if args.pooling:
        models['pooling'] = args.pooling
    if args.padding_side:
        models['padding_side'] = args.padding_side

    # Load architecture
    architecture = args.architecture or yaml_config.get('default_architecture_config')
    if architecture not in yaml_config['architectures']:
        print(f"error: architecture '{architecture}' not found")
        print(f"available architectures: {list(yaml_config['architectures'].keys())}")
        sys.exit(1)
    architectures = yaml_config['architectures'][architecture]

    if args.use_which_plan:
        architectures['use_which_plan'] = args.use_which_plan
    if args.output_layer is not None:
        architectures['output_layer'] = args.output_layer
    if args.tp_starting_index is not None:
        architectures['tp_starting_index'] = args.tp_starting_index
    if args.tp_exiting_index is not None:
        architectures['tp_exiting_index'] = args.tp_exiting_index

    # Apply GPU configuration
    if args.cuda_visible_devices:
        yaml_config['gpu_config']['cuda_visible_devices'] = args.cuda_visible_devices

    # Create a Config object
    cfg = Config(data=data, models=models, architectures=architectures)

    hyper_parameters = textwrap.dedent(f"""
        Configuration:
        -------------
        Model Method            : {model_method}
        Architecture            : {architecture}
        Pooling Method          : {models.get('pooling', 'N/A')}
        Padding Side            : {models.get('padding_side', 'N/A')}
        Prompt Method           : {cfg.models.template}
        Output Layer Index      : {architectures.get('output_layer', 'N/A')}
        Plan                    : {architectures.get('use_which_plan', 'N/A')}
        TP Starting layer Index : {architectures.get('tp_starting_index', 'N/A')}
        TP Exiting layer Index  : {architectures.get('tp_exiting_index', 'N/A')}
        Retrieval Datasets      : {data.get('tasks', 'N/A')}
    """)
    print(hyper_parameters)
    print(cfg)

    if 'qwen2' in architecture.lower():
        model = Qwen2ForCausalLM.from_pretrained(cfg.architecture.model_name_or_path,
                                                    device_map='auto',
                                                    output_hidden_states=True,
                                                    trust_remote_code=True)
        model.model.plan = cfg.architecture.use_which_plan
        model.model.tp_starting_index = cfg.architecture.tp_starting_index
        model.model.tp_exiting_index = cfg.architecture.tp_exiting_index
    elif 'gemma' in architecture.lower():
        model = GemmaForCausalLM.from_pretrained(cfg.architectures.model_name_or_path,
                                                    device_map='auto',
                                                    output_hidden_states=True,
                                                    trust_remote_code=True)
        model.model.plan = cfg.architectures.use_which_plan
        model.model.tp_starting_index = cfg.architectures.tp_starting_index
        model.model.tp_exiting_index = cfg.architectures.tp_exiting_index
    elif 'mistral' in architecture.lower():
        model = MistralForCausalLM.from_pretrained(cfg.architectures.model_name_or_path,
                                                    device_map='auto',
                                                    output_hidden_states=True,
                                                    trust_remote_code=True)
        model.model.plan = cfg.architectures.use_which_plan
        model.model.tp_starting_index = cfg.architectures.tp_starting_index
        model.model.tp_exiting_index = cfg.architectures.tp_exiting_index
    elif 'mistral' in architecture.lower():
        model = MistralForCausalLM.from_pretrained(cfg.architectures.model_name_or_path,
                                                    device_map='auto',
                                                    output_hidden_states=True,
                                                    trust_remote_code=True)
        model.model.plan = cfg.architectures.use_which_plan
        model.model.tp_starting_index = cfg.architectures.tp_starting_index
        model.model.tp_exiting_index = cfg.architectures.tp_exiting_index
    else:
        raise ValueError(f"Cannot find such {args.model_name_or_path.lower()} model!")
    
    print(f"Which model attention used: ", model.config._attn_implementation)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.architectures.model_name_or_path, add_bos_token=False, add_eos_token=False)
    tokenizer.pad_token_id = 0  # Set the padding token. we want this to be different from the eos token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<PST>", "<B-PST>"]})
    pst_token_id = tokenizer.convert_tokens_to_ids("<PST>")
    begin_pst_token_id = tokenizer.convert_tokens_to_ids("<B-PST>")
    print("<PST> Token ID:", pst_token_id)
    print("<B-PST> Token ID:", begin_pst_token_id)

    if cfg.architectures.use_which_plan != 'vanilla':
        placeholder_token = '<PST>'
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        begin_placeholder_token = '<B-PST>'
        begin_placeholder_token_id = tokenizer.convert_tokens_to_ids(begin_placeholder_token)

        model.resize_token_embeddings(len(tokenizer))
        
    sentence_process_function = create_add_pst_function(pst_token="<PST>", 
                                                        prepending_method = cfg.architectures.use_which_plan)
    parser = Parser(tokenizer=tokenizer,
                    templates=cfg.models.template, 
                    max_length=cfg.data.max_length,
                    prepending_method=cfg.architectures.use_which_plan,
                    padding_side=cfg.models.padding_side)
    
    pooling = Pooling(strategy=cfg.models.pooling,
                     padding_side=cfg.models.padding_side)

    embed_model = EmbeddingWrapper(model=model, 
                                   tokenizer=tokenizer,
                                   pooling = pooling,
                                   parser = parser,
                                   name=f"{model_method}-{architecture}-L:{cfg.architectures.output_layer}-{cfg.architectures.use_which_plan}",
                                   sentence_process_function=sentence_process_function,
                                   output_layer = cfg.architectures.output_layer,
                                   batch_size = cfg.data.batch_size,
                                   verbose = bool(args.verbose))

    tasks = cfg.data.tasks.split(',') if isinstance(cfg.data.tasks, str) else cfg.data.tasks

    name = f"{model_method}-{architecture}-L:{cfg.architectures.output_layer}-{cfg.architectures.use_which_plan}"
    results_dir = f"results/{name}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"results/{name}/sts_results.txt"  # File to save results

    # Clear the results file before writing
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("Retrieval Task Results\n")
            f.write("======================\n")

    for task in tasks:
        ds = load_dataset(f"mteb/{task}")['test']
        s1,s2,score = embed_model.encode(ds)
        cosine_scores = compute_cosine_similarities(s1, s2)
        correlation, p_value = spearmanr(cosine_scores.detach().cpu().numpy(), score.detach().cpu().numpy())
        print(f'evaluation results on dataset: {task},spearman corr: {correlation} with p-val: {p_value}')

         # Save task-specific results
        with open(results_file, "a+") as f:
            f.write(f"Task: {task}\n")
            f.write(f"spearman correlation Scores: {correlation}\n")
            f.write(f"P_value Scores: {p_value}\n")
            f.write("\n")
    

if __name__ == "__main__":
    main()