import argparse
import yaml
import sys
import textwrap
from model import Qwen2ForCausalLM, GemmaForCausalLM, MistralForCausalLM
from transformers import AutoTokenizer
import torch
torch.set_num_threads(24) 
from types import SimpleNamespace
from torch import nn
import numpy as np
from datasets import Dataset
from mteb.evaluation.evaluators.RetrievalEvaluator import PromptType
from datasets import Dataset
from torch.utils.data import DataLoader
import tqdm
from accelerate import Accelerator
from parser import Parser, create_add_pst_function
from pooling.pooling import Pooling
import mteb 
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import os
from evaluator import _compute_similarities_batched, retrieval_evaluate

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


def make_collate_functions(parser, key, prompt):
    def collate_fn(batch):
        if key == 'query':
            if prompt is not None:
                query_variables = [('query', {'prompt': prompt, 'text': q['texts']}) for q in batch]
            else:
                query_variables = [('query', {'text': q['texts']}) for q in batch]
            parsed = parser(query_variables)
            qids = [q['qid'] for q in batch]
            return {"qid": qids, "parsed": parsed}
        elif key == 'document':
            document_variables = [('document', {'text': d['texts']}) for d in batch]
            parsed = parser(document_variables)
            qids = [d['doc_id'] for d in batch]
            return {"qid": qids, "parsed": parsed}
    return collate_fn


def create_retrieval_dataloader(ds, 
                                parser, 
                                sentence_process_function=None,
                                query_key="query", 
                                document_key="document", 
                                prompt=None, 
                                query_batch_size=16,
                                document_batch_size=16):
    """
    Create a retrieval data loader for queries, corpus, and qrels.
    Args:
        ds: The dataset containing 'queries', 'corpus', and 'qrels'.
        parser: The parser object to process queries and documents.
        query_key: Key for query processing in the parser.
        document_key: Key for document processing in the parser.
        prompt: Optional prompt for query processing.
        batch_size: Batch size for the data loader.
    Returns:
        query_dataloader, document_dataloader, qrels_mapping
    """
    # Extract queries, corpus, and qrels
    queries = ds["queries"]
    corpus = ds["corpus"]
    qrels = ds["qrels"]

    # Create a mapping from qid to doc_ids for qrels
    qrels_mapping = defaultdict(list)
    for qrel in qrels:
        qrels_mapping[qrel["qid"]].append(qrel["doc_id"])
    
    accelerator = Accelerator()
    # Prepare query dataset
    if sentence_process_function is not None:
        print('Processing query datasets....')
        query_dataset = Dataset.from_list([{"texts": sentence_process_function(q["text"]), "qid": q["qid"]} for q in tqdm.tqdm(queries, desc="Processing queries")])
    else:
        query_dataset = Dataset.from_list([{"texts": q["text"], "qid": q["qid"]} for q in queries])
    query_collate_fn = make_collate_functions(parser, key=query_key, prompt=prompt)
    query_dataloader = DataLoader(query_dataset, batch_size=query_batch_size, collate_fn=query_collate_fn)
    query_dataloader = accelerator.prepare(query_dataloader)

    if sentence_process_function is not None:
        print('Processing document datasets....')
        document_dataset = Dataset.from_list([{"texts": sentence_process_function(d["text"]), "doc_id": d["doc_id"]} for d in tqdm.tqdm(corpus, desc="Processing documents")])
    else:
        document_dataset = Dataset.from_list([{"texts": d["text"], "doc_id": d["doc_id"]} for d in corpus])
    document_collate_fn = make_collate_functions(parser, key=document_key, prompt=None)
    document_dataloader = DataLoader(document_dataset, batch_size=document_batch_size, collate_fn=document_collate_fn)
    document_dataloader = accelerator.prepare(document_dataloader)
    return query_dataloader, document_dataloader, qrels_mapping



class RetrievalWrapper(nn.Module):
    def __init__(self, 
                 model, 
                 tokenizer,
                 pooling,
                 parser,
                 name=None,
                 prompt=None,
                 sentence_process_function=None,
                 query_batch_size=8,
                 document_batch_size=8,
                 output_layer = -2,
                 verbose = False):
        super(RetrievalWrapper, self).__init__()
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
        self.prompt= prompt
        self.query_batch_size = query_batch_size
        self.document_batch_size = document_batch_size

    
    def _do_encode_embeddings(self,
                              ds,
                              prompt=None):
        #print(variables)
        query_dataloader, document_dataloader, qrels_mapping = create_retrieval_dataloader(
                                                                        ds=ds,
                                                                        parser=self.parser,
                                                                        query_key="query",
                                                                        document_key="document",
                                                                        prompt=prompt,
                                                                        query_batch_size = self.query_batch_size,
                                                                        document_batch_size = self.document_batch_size,
                                                                        sentence_process_function=self.sentence_process_function
                                                                    )
        query_embeds = {}
        for batch in tqdm.tqdm(query_dataloader, desc='encoding', mininterval=10):
            qids = batch['qid'] 
            batch = batch['parsed']
            embed_mask = batch['embed_mask']
            input_ids = batch['input_ids']
            pst_positions = batch['pst_positions']
            eos_positions = batch['eos_positions']
            begin_pst_positions = batch['begin_pst_positions']
            if self.verbose:
                for idx, (pst_pos, eos_pos) in enumerate(zip(pst_positions, eos_positions)):
                    print(f"Example #{idx}")
                    print("PST positions:", pst_pos)
                    print("EOS positions:", eos_pos)
                    print("Begin PST positions:", begin_pst_positions[idx])
                    input_ids = batch["input_ids"][idx]
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
                    for pos in begin_pst_positions[idx]:
                        if pos < len(decoded_tokens):
                            print(f"  idx {pos}: {decoded_tokens[pos]}")
                    print("-" * 80)

            raw_outputs = self.model(**batch)
            hidden_states = raw_outputs.hidden_states
            outputs = self.pooling(token_embeddings = hidden_states,
                                   embed_mask = embed_mask)
            # Store embeddings in the dictionary with qids as keys
            for qid, embedding in zip(qids, outputs.detach().cpu().numpy()):
                query_embeds[qid] = embedding

        corpus_embeds = {}
        for batch in tqdm.tqdm(document_dataloader, desc='encoding', mininterval=10):
            qids = batch['qid'] 
            batch = batch['parsed']
            embed_mask = batch['embed_mask']
            input_ids = batch['input_ids']
            pst_positions = batch['pst_positions']
            eos_positions = batch['eos_positions']
            begin_pst_positions = batch['begin_pst_positions']

            if self.verbose:
                for idx, (pst_pos, eos_pos) in enumerate(zip(pst_positions, eos_positions)):
                    print(f"Example #{idx}")
                    print("PST positions:", pst_pos)
                    print("EOS positions:", eos_pos)
                    print("Begin PST positions:", begin_pst_positions[idx])
                    input_ids = batch["input_ids"][idx]
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
                    for pos in begin_pst_positions[idx]:
                        if pos < len(decoded_tokens):
                            print(f"  idx {pos}: {decoded_tokens[pos]}")
                    print("-" * 80)
            raw_outputs = self.model(**batch)
            hidden_states = raw_outputs.hidden_states
            outputs = self.pooling(token_embeddings = hidden_states,
                                   embed_mask = embed_mask)
            # Store embeddings in the dictionary with qids as keys
            for qid, embedding in zip(qids, outputs.detach().cpu().numpy()):
                corpus_embeds[qid] = embedding
        return query_embeds, corpus_embeds, qrels_mapping


    def evaluate(self, query_embeds, document_embeds, qrels_mapping):
        return retrieval_evaluate(query_embeds, document_embeds, qrels_mapping, top_k=100)


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
    data = yaml_config['data']['long_context_task']

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

    if cfg.architectures.use_which_plan != 'vanilla':
        tokenizer.add_special_tokens({"additional_special_tokens": ["<PST>", "<B-PST>"]})
        pst_token_id = tokenizer.convert_tokens_to_ids("<PST>")
        begin_pst_token_id = tokenizer.convert_tokens_to_ids("<B-PST>")
        print("<PST> Token ID:", pst_token_id)
        print("<B-PST> Token ID:", begin_pst_token_id)
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

    embed_model = RetrievalWrapper(model=model, 
                                   tokenizer=tokenizer,
                                   pooling = pooling,
                                   parser = parser,
                                   name=f"{model_method}-{architecture}-{cfg.architectures.use_which_plan}",
                                   prompt=None,
                                   sentence_process_function=sentence_process_function,
                                   output_layer = cfg.architectures.output_layer,
                                   query_batch_size=cfg.data.query_batch_size,
                                   document_batch_size=cfg.data.document_batch_size,
                                   verbose = bool(args.verbose))

    tasks = cfg.data.tasks.split(',') if isinstance(cfg.data.tasks, str) else cfg.data.tasks
    name = f"{model_method}-{architecture}-L:{cfg.architectures.output_layer}-{cfg.architectures.use_which_plan}"
    results_dir = f"results/{name}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"results/{name}/longembed_retrieval_results.txt"  # File to save results

    # Clear the results file before writing
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("Retrieval Task Results\n")
            f.write("======================\n")

    for task in tasks:
        print(f"Processing task: {task}")
        ds = mteb.load_dataset("dwzhu/LongEmbed",task)
        if ds is None:
            print(f"error: dataset {task} not found")
            continue

        with torch.no_grad():
            query_embeds, document_embeds, qrels_mapping = embed_model._do_encode_embeddings(ds)
            _, ndcg, _map, recall, precision = embed_model.evaluate(query_embeds, document_embeds, qrels_mapping)
            print(f"Results for task {task}: NDCG={ndcg}, MAP={_map}, Recall={recall}, Precision={precision}")


        # Save task-specific results
        with open(results_file, "a+") as f:
            f.write(f"Task: {task}\n")
            f.write(f"NDCG Scores: {dict(ndcg)}\n")
            f.write(f"MAP Scores: {dict(_map)}\n")
            f.write(f"Recall Scores: {dict(recall)}\n")
            f.write(f"Precision Scores: {dict(precision)}\n")
            f.write("\n")

    print(f"Results saved to {results_file}")
if __name__ == "__main__":
    main()