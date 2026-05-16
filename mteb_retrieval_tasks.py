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
                query_variables = [('query',{'prompt': prompt, 'text': q['texts']}) for q in batch]
            else:
                query_variables = [('query',{'text': q['texts']}) for q in batch]
            return parser(query_variables)
        elif key == 'document':
            document_variables= [('document', {'text': d['texts']}) for d in batch]
            return parser(document_variables)
    return collate_fn



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


    def encode(self, sentences, prompt_type=None, **kwargs):
        if prompt_type == PromptType.query:
            return self.encode_queries(sentences, **kwargs)
        elif prompt_type == PromptType.document:
            return self.encode_corpus(sentences, **kwargs)
        

    def encode_queries(self, sentences, **kwargs):
        return self._do_encode_embeddings(variables = sentences,
                                          key='query',
                                          batch_size=self.query_batch_size,
                                          prompt=self.prompt)


    def encode_corpus(self, sentences, **kwargs):
        return self._do_encode_embeddings(variables=sentences,
                                           key='document',
                                           prompt = None,
                                           batch_size=self.document_batch_size)
    
    
    def _do_encode_embeddings(self,
                              variables,
                              prompt=None,
                              key='query',
                              batch_size=16):
        #print(variables)
        if self.sentence_process_function is not None:
            variables = [self.sentence_process_function(t) for t in variables]
        #print(variables)
        dataset = Dataset.from_list([{"texts": t} for t in variables])
        collate_fn = make_collate_functions(self.parser, key, prompt)

        # Dynamic batch size adjustment based on actual sequence lengths
        def try_dataloader(bs):
            dataloader = DataLoader(dataset, batch_size=bs, collate_fn=collate_fn)
            accelerator = Accelerator()
            dataloader = accelerator.prepare(dataloader)
            max_seq_len = 0
            largest_batch = None
            
            # Find the batch with the largest sequence length
            for batch in dataloader:
                current_seq_len = batch['input_ids'].shape[1] * batch['input_ids'].shape[0]
                if current_seq_len > max_seq_len:
                    max_seq_len = current_seq_len
                    largest_batch = batch
            # Test with the largest batch to check for OOM
            if largest_batch is not None:
                try:
                    with torch.no_grad():
                        _ = self.model(output_hidden_states=True, **largest_batch)
                        return dataloader, bs
                except RuntimeError as e:
                    print(f"Error occurred: {e}")
                    if 'out of memory' in str(e):
                        print(f"Batch size {bs} is too large, trying smaller batch size...")
                        del largest_batch
                        torch.cuda.empty_cache()
                        return None, bs
                    else:
                        print(f"Unexpected error: {e}")
                        del largest_batch
                    torch.cuda.empty_cache()
                    return None, bs
 
        max_batch_size = batch_size
        min_batch_size = 1
        dataloader = None
        while max_batch_size >= min_batch_size:
            result = try_dataloader(max_batch_size)
            if result[0] is not None:
                dataloader = result[0]
                batch_size = result[1]
                print(f"Using batch size: {batch_size}")
                break
            print(f"Batch size {max_batch_size} failed, trying smaller batch size...")
            max_batch_size = max_batch_size // 2
        if dataloader is None:
            raise RuntimeError("Unable to fit batch into GPU memory, even with batch size 1.")

        encoded_embeds = []
        for batch in tqdm.tqdm(dataloader, desc='encoding', mininterval=10):
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
            raw_outputs = self.model(**batch,output_layer = self.output_layer)
            outputs = raw_outputs.hidden_states
            outputs = self.pooling(token_embeddings = outputs,
                                   embed_mask = embed_mask)
            encoded_embeds.append(outputs.detach().cpu().numpy())
        print(np.concatenate(encoded_embeds, axis=0).shape)
        return np.concatenate(encoded_embeds, axis=0)



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
    data = yaml_config['data']['retrieval_task']

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

    embed_model = RetrievalWrapper(model=model, 
                                   tokenizer=tokenizer,
                                   pooling = pooling,
                                   parser = parser,
                                   name=f"{model_method}-{architecture}-L:{cfg.architectures.output_layer}-{cfg.architectures.use_which_plan}",
                                   prompt=None,
                                   sentence_process_function=sentence_process_function,
                                   output_layer = cfg.architectures.output_layer,
                                   query_batch_size=cfg.data.query_batch_size,
                                   document_batch_size=cfg.data.document_batch_size,
                                   verbose = bool(args.verbose))

    tasks = cfg.data.tasks.split(',') if isinstance(cfg.data.tasks, str) else cfg.data.tasks
    tasks = mteb.get_tasks(tasks=tasks) # "NFCorpus", "FiQA2018"
    evaluation = mteb.MTEB(tasks=tasks)
    
    results = evaluation.run(embed_model, output_folder=f"results/{embed_model.name}",show_progress_bar=True)

if __name__ == "__main__":
    main()