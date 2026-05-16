import torch
import json
from templates import *
from data_loader import *
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from datasets import load_dataset
from torchmetrics.retrieval import RetrievalNormalizedDCG
from tqdm import tqdm
from collections import defaultdict
from accelerate import Accelerator
from scipy.stats import spearmanr
import os
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

def argument_parser():
    parser = argparse.ArgumentParser(description="sts embeddings")
    parser.add_argument('--pooling', type=str, default='mean', help='mean or last')
    parser.add_argument('--method', type=str, default='naiive',help='naiive, prompteol, echo, cot-prompteol, ke-prompteol')
    parser.add_argument('--hidden_layer', type=str, default='last',help='last, second_to_last')
    parser.add_argument('--base_model_path',type=str,default='mistralai/Mistral-7B-instruct-v0.3')
    parser.add_argument('--sts_datasets', nargs='+', help='List of STS datasets')
    parser.add_argument('--embedding_model', type= str, default='nvembed', help = 'embedding model')
    #parser.add_argument('--query_pos', type=str, default='query_0',help='query_0, query_1, query_2, query_3')
    args = parser.parse_args()
    return args



def make_sentence_collate_functions():
    def collate_fn(batch):
        grouped_batch = defaultdict(list)
        for d in batch:
            for k, v in d.items():
                grouped_batch[k].append(v)
        batch = dict(grouped_batch)
        score = [s for s in batch['score']]
        return { "s1": batch['sentence1'],
                "s2": batch['sentence2'],
                "score": score,
                }
    return collate_fn

def templates(method):
    template = {'echo': echo_templates,
     'naiive': naiive_templates,
     'prompteol':prompt_eol_templates,
     'cot-prompteol': cot_prompt_eol_templates,
     'cot-naiive': cot_naiive_templates,
     'cot-echo': cot_echo_templates}
    return template[method]

def compute_cosine_similarities(document_embeddings: torch.Tensor, 
                                      query_embeddings: torch.Tensor) -> torch.Tensor:
    document_normalized = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)
    query_normalized = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    return torch.diag(torch.matmul(query_normalized, document_normalized.T))


def main(args):
    #construct dataset
    print(args.sts_datasets)
    for dataset in args.sts_datasets:
        print(f'evaluating on dataset: {dataset}')
        ds_default = load_dataset(f"mteb/{dataset}")['test']
        if args.embedding_model == 'nvembed':
            model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', 
                                          trust_remote_code=True,
                                          device_map = "auto",
                                          torch_dtype= torch.float16)
        elif args.embedding_model =='qwen3':
            model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
                                        # model_kwargs={
                                        #             'torch_dtype': torch.float16,  # or torch.float16, torch.bfloat16
                                        #             'device_map': 'auto',   # or {'': 0} for specific GPU mapping
                                        #                 },
                                        # )#,torch_dtype= torch.float16,device_map = "auto")
        model = model.eval()

        collate_fn = make_sentence_collate_functions()
        accelerator = Accelerator()
        dataloader = DataLoader(ds_default, batch_size=2, shuffle=False, collate_fn = collate_fn, drop_last=False) 
        dataloader = accelerator.prepare(dataloader)
        s1_embeddings = []
        s2_embeddings = []
        scores = []
        max_length = 1024
        with torch.no_grad():
            for i in tqdm(dataloader):
                #print(i['s1'],i['s2'])
                if args.embedding_model == 'nvembed':
                    sentence_embeddings1 = model.encode(i['s1'], instruction="", max_length=max_length)
                    sentence_embeddings2 = model.encode(i['s2'], instruction="", max_length=max_length)
                elif args.embedding_model == 'qwen3':
                    sentence_embeddings1 = torch.from_numpy(model.encode(i['s1'], prompt="", max_length=max_length))
                    sentence_embeddings2 = torch.from_numpy(model.encode(i['s2'], prompt="", max_length=max_length))
                s1_embeddings.append(sentence_embeddings1)
                s2_embeddings.append(sentence_embeddings2)
                scores.append(torch.tensor(i['score'],dtype=torch.int))
        s1_embedding_result = torch.concat(s1_embeddings,dim=0)
        s2_embedding_result = torch.concat(s2_embeddings,dim=0)
        score_result = torch.concat(scores,dim=0)


        cosine_scores = compute_cosine_similarities(s1_embedding_result, s2_embedding_result)
        correlation, p_value = spearmanr(cosine_scores.detach().cpu().numpy(), score_result.detach().cpu().numpy())
        print(f'evaluation results on dataset: {dataset},spearman corr: {correlation} with p-val: {p_value}')

        # Create a results dictionary
        result_dict = {
            "dataset": dataset,
            "spearman_correlation": correlation,
            "p_value": p_value,
            "config": {
                "base_model_path": args.base_model_path,
                "pooling": args.pooling,
                "method": args.method,
                "hidden_layer": args.hidden_layer,
                "embedding_model": args.embedding_model
            }
        }

        # Create a folder to store results
        output_dir = "sts_results"
        os.makedirs(output_dir, exist_ok=True)

        # Create a result filename based on config
        result_file = os.path.join(
            output_dir,
            f"{dataset}__{args.method}__{args.pooling}__{args.hidden_layer}_embedding_model.json"
        )

        # Save to JSON
        with open(result_file, "a+") as f:
            json.dump(result_dict, f, indent=2)

        print(f"[✓] Saved results to {result_file}")


if __name__ == "__main__":
    args = argument_parser()
    main(args)