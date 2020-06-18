import torch.utils.data as tud
import torch
from typing import List
import random
import nlp


def prepare_dataset(tokenizer, split="train", max_length=120, num_datapoints=100_000):
    """Prepares WikiText-103 dataset"""
    wikitext = nlp.load_dataset("wikitext", "wikitext-103-v1")
    data = [x["text"] for x in wikitext[split]][:num_datapoints]
    data = "".join(data)
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data))
    chunked_token_ids = chunks(token_ids, max_length, tokenizer)
    data = Data(chunked_token_ids, tokenizer)
    return data


def chunks(lst, n, tokenizer):
    """Yield successive n-sized chunks from lst."""
    _chunks = []
    for i in range(0, len(lst), n):
        ids = [tokenizer.cls_token_id] + lst[i : i + n] + [tokenizer.sep_token_id]
        _chunks.append(torch.tensor(ids))
    return _chunks


def noise_text_input(text: str, noise_prob=0.2):
    """Takes a string, returns noised version of it"""
    splitted = text.split(" ")
    bool_mask = torch.empty(len(splitted)).uniform_() > 1 - noise_prob

    noised = []
    for word, boolean in zip(splitted, bool_mask):
        if boolean:
            if len(word) > 1:
                idx = random.randint(1, len(word) - 1)
                noised.append(word[:idx])
                noised.append(word[idx:])
        else:
            noised.append(word)
    return " ".join(noised)


def make_transformer_inputs(
    input_ids, max_length, padding_value, prefix="", make_labels=False, **kwargs
):
    lengths = [s.size(0) for s in input_ids]
    max_len = max(lengths)
    if max_len > max_length:
        max_len = max_length

    out_dims = (len(input_ids), max_len)
    padded_input_ids = input_ids[0].data.new(*out_dims).fill_(padding_value)
    attention_mask = padded_input_ids.clone()
    token_type_ids = padded_input_ids.clone()

    for i, tensor in enumerate(input_ids):
        length = tensor.size(0)
        if length > max_length:
            length = max_length
            tensor = tensor[:length]
        padded_input_ids[i, :length] = tensor
        attention_mask[i, :length] = torch.ones_like(tensor)

    batch = {
        f"{prefix}input_ids": padded_input_ids,
        f"{prefix}attention_mask": attention_mask,
        f"{prefix}token_type_ids": token_type_ids,
    }
    if make_labels:
        lm_labels = padded_input_ids.clone()
        lm_labels[lm_labels == padding_value] = -100
        batch["lm_labels"] = lm_labels

    batch.update(kwargs)
    return batch


class Data(tud.Dataset):
    def __init__(self, token_ids: List[torch.Tensor], tokenizer, noise_prob=0.2):
        self.token_ids = token_ids
        self.tokenizer = tokenizer
        self.len = len(token_ids)
        self.noise_prob = noise_prob

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tgt_ids = self.token_ids[idx]
        decoded = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
        noised = noise_text_input(decoded, self.noise_prob)
        src = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(noised))
        src = [self.tokenizer.cls_token_id] + src + [self.tokenizer.sep_token_id]
        src_ids = torch.tensor(src)
        return dict(src_input_ids=src_ids, tgt_input_ids=tgt_ids)


class Collater:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List):
        src = [x["src_input_ids"] for x in batch]
        tgt = [x["tgt_input_ids"] for x in batch]
        src_batch = self.collate(src)
        tgt_batch = self.collate(tgt, "decoder_", make_labels=True)
        src_batch.update(tgt_batch)
        return src_batch

    def collate(self, input_ids, prefix="", make_labels=False):
        return make_transformer_inputs(
            input_ids, self.max_length, self.tokenizer.pad_token_id, prefix, make_labels
        )
