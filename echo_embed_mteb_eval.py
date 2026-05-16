#this file is used to replicate echo embedding results on MTEB benchmark
import re
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from mteb.evaluation.evaluators.RetrievalEvaluator import PromptType
import mteb
from datasets import Dataset
from torch.utils.data import DataLoader
import tqdm
from accelerate import Accelerator


def make_collate_functions(parser, key, prompt):
    def collate_fn(batch):
        if key == 'query':
            query_variables = [('query',{'prompt': prompt, 'text': q['texts']}) for q in batch]
            return parser(query_variables)
        elif key == 'document':
            document_variables= [('document', {'text': d['texts']}) for d in batch]
            return parser(document_variables)
    return collate_fn

class EchoParser(nn.Module):
    def __init__(self, tokenizer, templates, max_length=None):
        super(EchoParser, self).__init__()
        self.tokenizer = tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, add_bos_token=False, add_eos_token=False)
        if self.tokenizer.padding_side != 'right':
            self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<unk>'
        self.templates = templates
        self.template_pieces = {k: self._parse_template(template) for k, template in templates.items()}
        self.max_length = max_length

    def _parse_template(self, template):
        matches = [m for m in re.finditer(r'\{(.|\n|\r|\t)+?\}', template)]
        template_pieces = []
        for i, m in enumerate(matches):
            if i == 0:
                template_pieces.append(template[:m.start()])
            else:
                template_pieces.append(template[matches[i - 1].end():m.start()])
            template_pieces.append(m.group())
        template_pieces.append(template[matches[-1].end():])
        template_pieces = [t for t in template_pieces if t]
        tokenized_pieces = []
        for template_piece in template_pieces:
            if template_piece.startswith('{') and template_piece.endswith('}'):
                tokenized_pieces.append(template_piece[1:-1])
            else:
                tokenized_pieces.append(self.tokenizer(template_piece)['input_ids'])
        return tokenized_pieces

    def _tokenize_piece(self, x, template_piece):
        if isinstance(template_piece, str):
            if template_piece.startswith('!'):
                template_piece = template_piece[1:]
                embed_mask_value = 0
            else:
                embed_mask_value = 1
            for k, v in x.items():
                template_piece = template_piece.replace(f'%%{k}%%', v)
            tokenized_piece = self.tokenizer(template_piece)['input_ids']
            tokenized_piece = tokenized_piece[:self.max_length] if self.max_length is not None else tokenized_piece
            attention_mask = torch.ones(len(tokenized_piece), dtype=torch.long)
            embed_mask = torch.full((len(tokenized_piece),), embed_mask_value, dtype=torch.long)
            return {
                'input_ids': torch.tensor(tokenized_piece, dtype=torch.long),
                'attention_mask': attention_mask,
                'embed_mask': embed_mask,
            }
        else:
            template_piece = template_piece[:self.max_length] if self.max_length is not None else template_piece
            attention_mask = torch.ones(len(template_piece), dtype=torch.long)
            embed_mask = torch.zeros(len(template_piece), dtype=torch.long)
            return {
                'input_ids': torch.tensor(template_piece, dtype=torch.long),
                'attention_mask': attention_mask,
                'embed_mask': embed_mask,
            }

    def _tokenize_from_pieces(self, x, template_pieces):
        token_pieces = [self._tokenize_piece(x, template_piece) for template_piece in template_pieces]
        tokens = {
            k: torch.cat([z[k] for z in token_pieces]) for k in token_pieces[0]
        }
        return tokens

    def tokenize(self, xs):
        # reminder: xs should ideally be a dict of str -> tuple(type, x)
        if isinstance(xs, tuple) and isinstance(xs[1], str):
            xs = [(xs[0], {'x': xs[1]})]
        elif isinstance(xs, tuple) and isinstance(xs[1], dict):
            xs = [xs]
        elif isinstance(xs[0], tuple) and isinstance(xs[0][1], str):
            xs = [(x[0], {'x': x[1]}) for x in xs]
        tokenized = [self._tokenize_from_pieces(x[1], self.template_pieces[x[0]]) for x in xs]
        max_tokenized_length = max([len(x['input_ids']) for x in tokenized])
        for x in tokenized:
            if len(x['input_ids']) < max_tokenized_length:
                x['input_ids'] = torch.cat([x['input_ids'], torch.tensor([self.tokenizer.pad_token_id] * (max_tokenized_length - len(x['input_ids'])), dtype=torch.long)])
                x['attention_mask'] = torch.cat([x['attention_mask'], torch.zeros(max_tokenized_length - len(x['attention_mask']), dtype=torch.long)])
                x['embed_mask'] = torch.cat([x['embed_mask'], torch.zeros(max_tokenized_length - len(x['embed_mask']), dtype=torch.long)])
        tokenized = {
            k: torch.stack([z[k] for z in tokenized]) for k in tokenized[0]
        }
        return tokenized

    def __call__(self, xs):
        tokens = self.tokenize(xs)
        return tokens

    def get_tokenizer(self):
        return self.tokenizer


class EchoPooling(nn.Module):
    def __init__(self, strategy='mean'):
        super(EchoPooling, self).__init__()
        self.strategy = strategy
    
    def forward(self, xs):
        token_embeddings = xs['token_embeddings']
        embed_mask = xs['embed_mask'].to(token_embeddings.device)        
        if self.strategy == 'mean':
            pooled = torch.sum(token_embeddings * embed_mask.unsqueeze(-1), dim=1) / torch.sum(embed_mask, dim=1).unsqueeze(-1)
            pooled.masked_fill_(torch.isnan(pooled), 0)
        elif self.strategy == 'last':
            def _extract_last_nonzero(m):
                nonzeros = (m == 1).nonzero(as_tuple=True)[0]
                return torch.max(nonzeros) if nonzeros.size(0) > 0 else 0
            last_indices = torch.tensor([_extract_last_nonzero(m) for m in embed_mask])
            i = torch.arange(token_embeddings.shape[0]).reshape(token_embeddings.shape[0], 1, 1)
            j = last_indices.reshape(last_indices.shape[0], 1, 1)
            k = torch.arange(token_embeddings.shape[2])
            pooled = token_embeddings[i, j, k][:, 0, :]
            pooled.masked_fill_(torch.isnan(pooled), 0)
        else:
            raise ValueError(f'Unknown pooling strategy: {self.strategy}')
        xs.update({
            'sentence_embedding': pooled,
        })
        return xs


class EchoEmbeddingsMistral(nn.Module):
    def __init__(self, model, parser=None, pooling=None):
        super(EchoEmbeddingsMistral, self).__init__()
        self.model = model
        self.parser = parser
        self.pooling = pooling

    def forward(self, xs):
        inputs = {
            'input_ids': xs['input_ids'].to(self.model.device),
            'attention_mask': xs['attention_mask'].to(self.model.device),
        }
        outputs = self.model(**inputs).last_hidden_state
        xs.update({
            'token_embeddings': outputs,
        })
        return xs

    @staticmethod
    def from_pretrained(base_model_path, **kwargs):
        base_model = AutoModel.from_pretrained(base_model_path, **kwargs)
        return EchoEmbeddingsMistral(base_model)


class EmbeddingModel(nn.Module):
    def __init__(self, model, name=None):
        super(EmbeddingModel, self).__init__()
        self.model = model
        self.name = None

    @staticmethod
    def from_pretrained(base_model_path, **kwargs):
        base_model = AutoModel.from_pretrained(base_model_path, **kwargs)
        return EmbeddingModel(base_model)
    

    def encode(self, sentences, prompt_type=None, **kwargs):
        print(prompt_type)
        if prompt_type == PromptType.query:
            return self.encode_queries(sentences, **kwargs)
        elif prompt_type == PromptType.document:
            return self.encode_corpus(sentences, **kwargs)
    

    def encode_queries(self, sentences, **kwargs):
        prompt= 'Given a scientific claim, retrieve documents that support or refute the claim'
        return self._do_encode_embeddings(sentences,prompt,'query',batch_size=16)


    def encode_corpus(self, sentences, **kwargs):
        return self._do_encode_embeddings(sentences,'','document',batch_size=16)
    
    
    def _do_encode_embeddings(self,variables,prompt=None,key='query',batch_size=16,max_length=512):
        dataset = Dataset.from_dict({"texts": variables})
        # Create the DataLoader
        collate_fn = make_collate_functions(self.model.parser,key,prompt)
        dataloader = DataLoader(dataset,   
                                batch_size=batch_size,
                                collate_fn=collate_fn
                            )
        accelerator = Accelerator()
        dataloader = accelerator.prepare(dataloader)
        encoded_embeds = []
        for val in tqdm.tqdm(dataloader, desc='encoding', mininterval=10):
            # Decode and print the input_ids
            decoded_texts = self.model.parser.get_tokenizer().batch_decode(val['input_ids'], skip_special_tokens=False)
            #print("Decoded texts:", decoded_texts)
            sentence_embeddings =  self.model.pooling(self.model.forward(val))['sentence_embedding']
            encoded_embeds.append(sentence_embeddings.detach().cpu().numpy())
        print(np.concatenate(encoded_embeds, axis=0).shape)
        return np.concatenate(encoded_embeds, axis=0)









echo_retrieval_templates = {
    'query': '<s>Instruct:{!%%prompt%%,}\nQuery:\"{!%%text%%}\"\nQuery again:\"{%%text%%}\"{</s>}',
    'document': '<s>Document:\"{!%%text%%}\"\nDocument again:\"{%%text%%}\"{</s>}',
}

echo_model =EchoEmbeddingsMistral(model = AutoModel.from_pretrained('mistralai/Mistral-7B-instruct-v0.1',
                                                               device_map="auto",
                                                               output_hidden_states=True),
                             parser= EchoParser('mistralai/Mistral-7B-instruct-v0.1',
                                                echo_retrieval_templates,
                                                max_length=512),
                             pooling=EchoPooling(strategy='mean'))

model = EmbeddingModel(echo_model)
model.eval()
model.name = "echoembed"
with torch.no_grad():
    print(getattr(model, 'name', 'no_model_name'))
    tasks = mteb.get_tasks(tasks=["SciFact","NFCorpus","FiQA2018",'SCIDOCS'])#, "NFCorpus", "FiQA2018"])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=f"results/echoembed2",show_progress_bar=True)
    print(results[0].scores['test'][0]['ndcg_at_10'])#['test'][0]["ndcg_at_10"])