import re
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import mteb
import tqdm
import re
import spacy


nlp = spacy.load('en_core_web_sm')

def create_add_pst_function(pst_token="<PST>", prepending_method='vanilla'):
    def add_pst(paragraph):
        return add_pst_to_sentences(paragraph, pst_token=pst_token, prepending_method=prepending_method)
    return add_pst


def create_add_pst_every_k_tokens_function(k, tokenizer, pst_token="<PST>"):
    def add_pst(paragraph):
        return add_pst_every_k_tokens_with_tokenizer(paragraph, k, tokenizer, pst_token=pst_token)
    return add_pst


def add_pst_to_sentences(paragraph,
                        prepending_method='vanilla', 
                         pst_token="<PST>"):
    if prepending_method == 'vanilla' or prepending_method =='tp':
        return paragraph
    doc = nlp(paragraph.strip())
    sentences = []
    for sent in doc.sents:
        s = sent.text.strip()
        if not s.endswith(('.', '!', '?',"\"")):
            s += '.'
        sentences.append(f"{pst_token} {s}")
    if sentences == []:
        return f"{pst_token} ."
    return ' '.join(sentences)


def add_pst_every_k_tokens_with_tokenizer(paragraph, k, tokenizer, pst_token="<PST>"):
    # Tokenize the paragraph
    tokens = tokenizer(paragraph, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    # Split tokens into chunks of size k
    chunks = [tokens[i:i + k] for i in range(0, len(tokens), k)]
    # Add <PST> token ID to the beginning of each chunk
    pst_token_id = tokenizer.convert_tokens_to_ids(pst_token)
    modified_chunks = [torch.cat((torch.tensor([pst_token_id]), chunk)) for chunk in chunks]
    # Concatenate all chunks back into a single tensor
    modified_tokens = torch.cat(modified_chunks)
    # Decode the modified tokens back into text
    modified_paragraph = tokenizer.decode(modified_tokens, skip_special_tokens=False)
    return modified_paragraph


#Adapted from Echo Eembedding's Parser
class Parser(nn.Module):
    def __init__(self, 
                 tokenizer, 
                 templates, 
                 prepending_method = 'vanilla',
                 padding_side = 'right',
                 max_length=None):
        super(Parser, self).__init__()
        self.tokenizer = tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, add_bos_token=False, add_eos_token=False)
        self.templates = templates
        self.template_pieces = {k: self._parse_template(template) for k, template in templates.items()}
        self.max_length = max_length
        self.add_prepending = False
        self.add_begin_prepending = False
        self.omit_pst = False
        self.add_sentence_prepending = False
        if 'tp' in prepending_method:
            self.pst_token_id = self.tokenizer.convert_tokens_to_ids("<PST>")
            self.dot_id = self.tokenizer.convert_tokens_to_ids(".")
            self.excl_id = self.tokenizer.convert_tokens_to_ids("!")
            self.quest_id = self.tokenizer.convert_tokens_to_ids("?")
            self.add_prepending = True
            if 'tp_sentence' in prepending_method:
                self.add_sentence_prepending = True
            if prepending_method == 'tp_sentence_begin' or prepending_method == 'tp_sentence_omit_pst':
                self.begin_pst_token_id = self.tokenizer.convert_tokens_to_ids("<B-PST>")
                self.add_begin_prepending = True
            if prepending_method == 'tp_sentence_omit_pst':
                self.omit_pst = True
        self.padding_side = padding_side

    def _extract_last_nonzero(self,m):
        nonzeros = (m == 1).nonzero(as_tuple=True)[0]
        return torch.max(nonzeros) if nonzeros.size(0) > 0 else 0
            

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

            # Tokenize the template piece
            tokenized_piece = self.tokenizer(template_piece)['input_ids']
            tokenized_piece = tokenized_piece[:self.max_length] if self.max_length is not None else tokenized_piece
            attention_mask = torch.ones(len(tokenized_piece), dtype=torch.long)
            embed_mask = torch.full((len(tokenized_piece),), embed_mask_value, dtype=torch.long)
            

            if  self.add_begin_prepending:
                # Count <PST> tokens in the truncated tokenized_piece
                num_pst_tokens = sum(1 for tid in tokenized_piece if tid == self.pst_token_id)
                # Add <B-PST> tokens to the beginning
                if num_pst_tokens > 0:
                    tokenized_piece = [self.begin_pst_token_id] * num_pst_tokens + tokenized_piece
                # Create attention_mask and embed_mask
                attention_mask = torch.ones(len(tokenized_piece), dtype=torch.long)
                embed_mask = torch.full((len(tokenized_piece),), embed_mask_value, dtype=torch.long)
                # Set embed_mask to 0 for <PST> and <B-PST> token locations
                pst_indices = [i for i, tid in enumerate(tokenized_piece) if tid == self.pst_token_id or tid == self.begin_pst_token_id]
                if pst_indices:
                    embed_mask[pst_indices] = 0

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
        text_piece_idx = None
        for idx, piece in enumerate(template_pieces):
            if isinstance(piece, str) and '%%text%%' in piece:
                text_piece_idx = idx
                break
        assert text_piece_idx is not None, "%%text%% not found in template_pieces!"
        piece_lengths = [len(p['input_ids']) for p in token_pieces]
        text_end_idx = sum(piece_lengths[:text_piece_idx + 1])

        if self.add_prepending:
            if not self.add_sentence_prepending:
                input_ids = tokens['input_ids']
                pst_positions = (input_ids == self.pst_token_id).nonzero(as_tuple=True)[0].tolist()
                if self.padding_side == 'right':
                    eos_positions = [self._extract_last_nonzero(tokens['embed_mask']).tolist()]
                elif self.padding_side == 'left':
                    eos_positions = [[-1] * len(tokens['embed_mask'])]
                tokens['begin_pst_positions'] = []
                tokens['pst_positions'] = pst_positions
                tokens['eos_positions'] = eos_positions
            if self.add_sentence_prepending:
                input_ids = tokens['input_ids']
                pst_positions = (input_ids == self.pst_token_id).nonzero(as_tuple=True)[0].tolist()
                eos_positions = [i-1 for i in pst_positions[1:]]
                eos_positions.append(text_end_idx)
                tokens['pst_positions'] = pst_positions
                tokens['eos_positions'] = eos_positions
                tokens['begin_pst_positions'] = []
                if self.add_begin_prepending:
                    if self.omit_pst:
                        # Remove all <PST> tokens from input_ids, attention_mask, and embed_mask
                        eos_tensor = torch.zeros_like(input_ids, dtype=torch.float)
                        eos_tensor[eos_positions] = 1
                        pst_mask = input_ids != self.pst_token_id
                        input_ids = input_ids[pst_mask]
                        tokens['input_ids'] = input_ids
                        tokens['attention_mask'] = tokens['attention_mask'][pst_mask]
                        tokens['embed_mask'] = tokens['embed_mask'][pst_mask]
                        eos_tensor = eos_tensor[pst_mask]
                        begin_pst_positions = (input_ids == self.begin_pst_token_id).nonzero(as_tuple=True)[0].tolist()
                        tokens['pst_positions'] = begin_pst_positions
                        tokens['eos_positions'] = eos_tensor.nonzero(as_tuple=True)[0].tolist()
                    else:
                        begin_pst_positions = (input_ids == self.begin_pst_token_id).nonzero(as_tuple=True)[0].tolist()
                        tokens['begin_pst_positions'] = begin_pst_positions
        else:
            tokens['pst_positions'] = []
            tokens['eos_positions'] = []
            tokens['begin_pst_positions'] = []
        return tokens 



    def tokenize(self, xs):
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
                if self.padding_side == 'right':
                    x['input_ids'] = torch.cat([x['input_ids'], torch.tensor([self.tokenizer.pad_token_id] * (max_tokenized_length - len(x['input_ids'])), dtype=torch.long)])
                    x['attention_mask'] = torch.cat([x['attention_mask'], torch.zeros(max_tokenized_length - len(x['attention_mask']), dtype=torch.long)])
                    x['embed_mask'] = torch.cat([x['embed_mask'], torch.zeros(max_tokenized_length - len(x['embed_mask']), dtype=torch.long)])
                elif self.padding_side == 'left':
                    pad_len = max_tokenized_length - len(x['input_ids'])
                    x['input_ids'] = torch.cat([torch.tensor([self.tokenizer.pad_token_id] * (max_tokenized_length - len(x['input_ids'])), dtype=torch.long), x['input_ids']])
                    x['attention_mask'] = torch.cat([torch.zeros(max_tokenized_length - len(x['attention_mask']), dtype=torch.long), x['attention_mask']])
                    x['embed_mask'] = torch.cat([torch.zeros(max_tokenized_length - len(x['embed_mask']), dtype=torch.long), x['embed_mask']])
                    x['pst_positions'] = [pos + pad_len for pos in x['pst_positions']]
                    x['eos_positions'] = [pos + pad_len for pos in x['eos_positions']]
                    x['begin_pst_positions'] = [pos + pad_len for pos in x['begin_pst_positions']]
        tensor_keys = ['input_ids', 'attention_mask', 'embed_mask']
        combined = {
            k: torch.stack([z[k] for z in tokenized]) if k in tensor_keys else [z[k] for z in tokenized]
            for k in tokenized[0]
        }
        return combined


    def __call__(self, xs):
        tokens = self.tokenize(xs)
        return tokens


    def get_tokenizer(self):
        return self.tokenizer