# from abc import ABCMeta, abstractmethod
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import pyfaidx
# from scipy.stats import wilcoxon
# from tqdm import tqdm

from ..utils import one_hot_encode, copy_if_not_exists, onehot_to_chars

class PairedControlDataset(Dataset):
    _elements_dtypes = {
        "chr": pl.Utf8,
        "input_start": pl.UInt32,
        "input_end": pl.UInt32,
        "ccre_start": pl.UInt32,
        "ccre_end": pl.UInt32,
        "ccre_relative_start": pl.Int32,
        "ccre_relative_end": pl.Int32,
        "reverse_complement": pl.Boolean
    }

    _seq_tokens = np.array([0, 1, 2, 3], dtype=np.int8)

    _seed_upper = 2**128
    
    
    def __init__(self, genome_fa, elements_tsv, chroms, seed, tokenizer, cache_token=True):
        super().__init__()
        self.seed = seed
        self.tokenizer = tokenizer
        self.cache_token = cache_token
        self.elements_df = self._load_elements(elements_tsv, chroms)
        self.genome_fa = genome_fa
        
        self.token_cache = {} if self.cache_token else None  # {idx: (seq_tokens, ctrl_tokens)}

    def __len__(self):
        return self.elements_df.height

    def __getitem__(self, idx):
        if self.cache_token and idx in self.token_cache:
            return self.token_cache[idx] + (torch.tensor(idx),)

        # Load metadata
        fa = pyfaidx.Fasta(self.genome_fa, one_based_attributes=False)
        idx_orig, chrom, start, end, elem_start, elem_end, _, _, rc = self.elements_df.row(idx)
        window = end - start

        # Get deterministic seed for reproducibility
        item_bytes = (self.seed, chrom, elem_start, elem_end).__repr__().encode('utf-8')
        item_seed = int(hashlib.sha256(item_bytes).hexdigest(), 16) % (2**128)
        rng = np.random.default_rng(item_seed)

        # Fetch sequence
        seq = np.zeros((window, 4), dtype=np.int8)
        sequence_data = fa[chrom][max(0, start):end]
        sequence = sequence_data.seq.upper()
        a = sequence_data.start - start
        b = sequence_data.end - start
        seq[a:b, :] = one_hot_encode(sequence)

        # Get element & generate shuffled control
        e_a = max(elem_start - start, a)
        e_b = min(elem_end - start, b)
        elem = seq[e_a:e_b, :]
        shuf = self._dinuc_shuffle(elem, rng)
        ctrl = seq.copy()
        ctrl[e_a:e_b, :] = shuf

        if rc:
            seq = seq[::-1, ::-1].copy()
            ctrl = ctrl[::-1, ::-1].copy()

        # Convert to tokens
        seq_str = onehot_to_chars(seq)
        ctrl_str = onehot_to_chars(ctrl)

        seq_token = self.tokenizer(seq_str, return_tensors="pt", padding="longest", truncation=True)
        ctrl_token = self.tokenizer(ctrl_str, return_tensors="pt", padding="longest", truncation=True)

        seq_tokens = seq_token["input_ids"].squeeze(0)
        ctrl_tokens = ctrl_token["input_ids"].squeeze(0)

        if self.cache_token:
            self.token_cache[idx] = (seq_tokens, ctrl_tokens)

        fa.close()
        return seq_tokens, ctrl_tokens, torch.tensor(idx)

    @classmethod
    def _load_elements(cls, elements_file, chroms):
        df = pl.scan_csv(elements_file, separator="\t", quote_char=None, dtypes=cls._elements_dtypes).with_row_index()
        if chroms is not None:
            df = df.filter(pl.col("chr").is_in(chroms))
        return df.collect()

    @classmethod
    def _dinuc_shuffle(cls, seq, rng):
        tokens = (seq * cls._seq_tokens[None, :]).sum(axis=1)
        shuf_next_inds = []
        for t in range(4):
            inds = np.where(tokens[:-1] == t)[0]
            shuf_next_inds.append(inds + 1)
        for t in range(4):
            inds = np.arange(len(shuf_next_inds[t]))
            if len(inds) > 1:
                inds[:-1] = rng.permutation(len(inds) - 1)
            shuf_next_inds[t] = shuf_next_inds[t][inds]
        counters = [0] * 4
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]
        return (result[:, None] == cls._seq_tokens[None, :]).astype(np.int8)
