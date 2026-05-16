import argparse
import yaml
import sys
import textwrap
from transformers import AutoTokenizer, AutoModel
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
from torch import Tensor

class Config:
    def __init__(self, data=None, models=None, architectures=None):
        self.data = SimpleNamespace(**(data or {}))
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


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def make_collate_functions(tokenizer, key, prompt):
    def collate_fn(batch):
        if key == 'query':
            if prompt is not None:
                query_variables = [get_detailed_instruct(query['text']) for query in batch]
            else:
                query_variables = [query['text'] for query in batch]
            parsed = tokenizer(query_variables)
            qids = [q['qid'] for q in batch]
            return {"qid": qids, "parsed": parsed}
        elif key == 'document':
            document_variables = [document['text'] for document in batch]
            parsed = tokenizer(document_variables)
            qids = [d['doc_id'] for d in batch]
            return {"qid": qids, "parsed": parsed}
    return collate_fn


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]



def create_retrieval_dataloader(ds, 
                                tokenizer, 
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

    query_dataset = Dataset.from_list([{"texts": q["text"], "qid": q["qid"]} for q in queries])
    query_collate_fn = make_collate_functions(tokenizer, key=query_key, prompt=prompt)
    query_dataloader = DataLoader(query_dataset, batch_size=query_batch_size, collate_fn=query_collate_fn)
    query_dataloader = accelerator.prepare(query_dataloader)

    document_dataset = Dataset.from_list([{"texts": d["text"], "doc_id": d["qid"]} for d in corpus])
    document_collate_fn = make_collate_functions(tokenizer, key=document_key, prompt=None)
    document_dataloader = DataLoader(document_dataset, batch_size=document_batch_size, collate_fn=document_collate_fn)
    document_dataloader = accelerator.prepare(document_dataloader)
    return query_dataloader, document_dataloader, qrels_mapping



class RetrievalWrapper(nn.Module):
    def __init__(self, 
                 model, 
                 tokenizer,
                 pooling,
                 name=None,
                 prompt=None,
                 query_batch_size=8,
                 document_batch_size=8):
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
        self.prompt= prompt
        self.query_batch_size = query_batch_size
        self.document_batch_size = document_batch_size
        self.pooling = pooling
    
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
                                                                        document_batch_size = self.document_batch_size
                                                                    )
        query_embeds = {}
        for batch in tqdm.tqdm(query_dataloader, desc='encoding', mininterval=10):
            qids = batch['qid'] 
            batch = batch['parsed']
            outputs = self.model(**batch)
            outputs = self.pooling(outputs.last_hidden_state, batch['attention_mask'])
            for qid, embedding in zip(qids, outputs.detach().cpu().numpy()):
                query_embeds[qid] = embedding

        corpus_embeds = {}
        for batch in tqdm.tqdm(document_dataloader, desc='encoding', mininterval=10):
            qids = batch['qid'] 
            batch = batch['parsed']
            outputs = self.model(**batch)
            outputs = self.pooling(outputs.last_hidden_state, batch['attention_mask'])
            for qid, embedding in zip(qids, outputs.detach().cpu().numpy()):
                corpus_embeds[qid] = embedding
        return query_embeds, corpus_embeds, qrels_mapping

    def evaluate(self, query_embeds, document_embeds, qrels_mapping):
        return retrieval_evaluate(query_embeds, document_embeds, qrels_mapping, top_k=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default=None,
                        help="Select which architecture to use")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Change model_name_or_path in architectures")
    parser.add_argument("--use_which_plan", type=str, default=None,
                        help="Change use_which_plan in architectures")
    parser.add_argument("--retrieval_datasets", type=str, nargs='+', default=None,
                        help="Pass a list of datasets for retrieval_task")
    args = parser.parse_args()
    yaml_config = load_config_from_yaml()

    # Load data task
    data = yaml_config['data']['long_context_task']

    # Override retrieval datasets if provided
    if args.retrieval_datasets:
        data['tasks'] = ','.join(args.retrieval_datasets)

    # Load architecture
    architecture = args.architecture or yaml_config.get('default_architecture_config')
    if architecture not in yaml_config['architectures']:
        print(f"error: architecture '{architecture}' not found")
        print(f"available architectures: {list(yaml_config['architectures'].keys())}")
        sys.exit(1)
    architectures = yaml_config['architectures'][architecture]

    # Apply GPU configuration
    if args.cuda_visible_devices:
        yaml_config['gpu_config']['cuda_visible_devices'] = args.cuda_visible_devices

    # Create a Config object
    cfg = Config(data=data, architectures=architectures)

    hyper_parameters = textwrap.dedent(f"""
        Configuration:
        -------------
        Architecture            : {architecture}
        Prompt Method           : {cfg.models.template}
        Retrieval Datasets      : {data.get('tasks', 'N/A')}
    """)
    print(hyper_parameters)

    if 'qwen3' in architecture.lower():
        tokenizer = AutoTokenizer.from_pretrained(architectures.model_name_or_path, padding_side='left')
        model = AutoModel.from_pretrained(architectures.model_name_or_path)
        pooling_func = last_token_pool
    elif 'e5' in architecture.lower():
        tokenizer = AutoTokenizer.from_pretrained(architectures.model_name_or_path)
        model = AutoModel.from_pretrained(architectures.model_name_or_path)
        pooling_func = last_token_pool
    else:
        raise ValueError(f"Cannot find such {args.model_name_or_path.lower()} model!")
    

    embed_model = RetrievalWrapper(model=model, 
                                   tokenizer=tokenizer,
                                   pooling = last_token_pool,
                                   name=f"{architecture}",
                                   prompt=None,
                                   query_batch_size=cfg.data.query_batch_size,
                                   document_batch_size=cfg.data.document_batch_size
                                   )

    tasks = cfg.data.tasks.split(',') if isinstance(cfg.data.tasks, str) else cfg.data.tasks
    name = f"{architecture}"
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