from abc import ABCMeta, abstractmethod
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM, BertConfig
from scipy.stats import wilcoxon
from tqdm import tqdm

from ..components import PairedControlDataset
from ...utils import onehot_to_chars, NoModule

class MaskedZeroShotScore(metaclass=ABCMeta):
    @property
    @abstractmethod
    def mask_token(self):
        pass

    def score(self, tokens, starts, ends, attention_mask):
        tokens = tokens.to(device=self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
        lls = torch.zeros(tokens.shape[:2], device=self.device)
        for i in range(tokens.shape[1]):
            clip_mask = ((i >= starts) & (i < ends)).to(device=self.device)
            masked_tokens = tokens.clone()
            masked_tokens[:,i,...] = self.mask_token
            lls[:,i] = self.model_fwd(masked_tokens, attention_mask, tokens)[:,i] * clip_mask

        out = lls.sum(dim=1).numpy(force=True)

        return out
    

class CausalZeroShotScore(metaclass=ABCMeta):
    def score(self, tokens, starts, ends, attention_mask):
        tokens = tokens.to(device=self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
        lls = self.model_fwd(tokens, attention_mask, tokens)
        clip_mask = torch.zeros_like(lls)
        for i in range(lls.shape[1]):
            clip_mask[:,i] = ((i >= starts) & (i < ends))

        # clip_mask = torch.tensor([[(i >= s) and (i < e) for i in range(lls.shape[1])] for s, e in zip(starts, ends)], 
        #                          dtype=torch.float).to(device=self.device)

        out = (lls * clip_mask).sum(1).numpy(force=True)

        return out
    
    
 
import torch.multiprocessing as mp
import logging
import os
import time
import traceback

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S"
)

class ZeroShotPairedControlEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, dataset, batch_size, num_workers, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def tokenize(self, seqs):
        """Should return tokens, starts, ends, attention_mask"""
        pass

    @abstractmethod
    def score(self, tokens, starts, ends, attention_mask):
        """Compute and return scores for a batch"""
        pass

    def evaluate(self, out_dir, progress_bar=False):
        os.makedirs(out_dir, exist_ok=True)
        # ✅ 获取可用 GPU 数量并拆分数据
        num_gpus = torch.cuda.device_count()
        chunks = torch.utils.data.random_split(
            self.dataset,
            [len(self.dataset) // num_gpus + int(i < len(self.dataset) % num_gpus) for i in range(num_gpus)]
        )

        ctx = mp.get_context("spawn")
        manager = mp.Manager()
        # ✅ 共享列表用于存储各进程结果
        shared_results = manager.list([None] * num_gpus)

        processes = []
        # 启动所有 GPU 进程
        for rank in range(num_gpus):
            part_path = os.path.join(out_dir, f"scores_part{rank}.tsv")
            p = ctx.Process(
                target=self._evaluate_worker,
                args=(chunks[rank], rank, part_path, shared_results, progress_bar)
            )
            p.start()
            processes.append(p)

        # 等待所有进程完成
        for p in processes:
            p.join()

        # 合并所有结果
        all_inds = []
        all_seq_scores = []
        all_ctrl_scores = []
        for rank in range(num_gpus):
            result = shared_results[rank]
            if result is None:
                logging.error(f"[Main] GPU {rank} returned no results.")
                inds, seq_s, ctrl_s = [], [], []
            else:
                inds, seq_s, ctrl_s = result
            all_inds.extend(inds)
            all_seq_scores.extend(seq_s)
            all_ctrl_scores.extend(ctrl_s)

        # 写入合并后的 scores.tsv
        scores_path = os.path.join(out_dir, "scores.tsv")
        with open(scores_path, "w") as f:
            f.write("idx\tseq_score\tctrl_score\n")
            # 可选：按索引排序
            for ind, s, c in sorted(zip(all_inds, all_seq_scores, all_ctrl_scores), key=lambda x: x[0]):
                f.write(f"{ind}\t{s}\t{c}\n")

        # 计算指标
        diffs = np.array(all_seq_scores) - np.array(all_ctrl_scores)
        corrects = diffs > 0

        metrics = {
            "acc": float(corrects.mean())
        }
        wilcox = wilcoxon(diffs, alternative="greater")
        metrics.update({
            "pval": float(wilcox.pvalue),
            "signed_rank_sum": float(wilcox.statistic),
            "mean_diff": float(diffs.mean()),
            "q05_diff": float(np.percentile(diffs, 5)),
            "q25_diff": float(np.percentile(diffs, 25)),
            "median_diff": float(np.median(diffs)),
            "q75_diff": float(np.percentile(diffs, 75)),
            "q95_diff": float(np.percentile(diffs, 95))
        })

        # 写入 metrics.json
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics

    # def _evaluate_worker(self, dataset_split, gpu_id, output_file, shared_results, progress_bar):
    #     try:
    #         # 设置设备
    #         torch.cuda.set_device(gpu_id)
    #         self.device = torch.device(f"cuda:{gpu_id}")
    #         self.model.to(self.device)

    #         dataloader = DataLoader(dataset_split, batch_size=self.batch_size, shuffle=False, num_workers=0)

    #         inds_list = []
    #         seq_scores_list = []
    #         ctrl_scores_list = []

    #         # 写入分块结果
    #         with open(output_file, "w") as f:
    #             f.write("idx\tseq_score\tctrl_score\n")
    #             pbar = tqdm(dataloader,
    #                         disable=(not progress_bar), ncols=120,
    #                         position=gpu_id, desc=f"[GPU {gpu_id}]")
    #             for batch_idx, (seqs, ctrls, inds) in enumerate(pbar):
    #                 try:
    #                     seq_tokens, seq_starts, seq_ends, seq_mask = self.tokenize(seqs)
    #                     ctrl_tokens, ctrl_starts, ctrl_ends, ctrl_mask = self.tokenize(ctrls)

    #                     seq_scores = self.score(seq_tokens, seq_starts, seq_ends, seq_mask)
    #                     ctrl_scores = self.score(ctrl_tokens, ctrl_starts, ctrl_ends, ctrl_mask)

    #                     for ind, s_score, c_score in zip(inds.tolist(), seq_scores.flatten(), ctrl_scores.flatten()):
    #                         inds_list.append(ind)
    #                         seq_scores_list.append(float(s_score))
    #                         ctrl_scores_list.append(float(c_score))
    #                         f.write(f"{ind}\t{s_score}\t{c_score}\n")
    #                     f.flush()
    #                 except Exception as batch_err:
    #                     logging.error(f"[GPU {gpu_id}] Error in batch {batch_idx}: {batch_err}")
    #                     logging.error(traceback.format_exc())

    #         logging.info(f"[GPU {gpu_id}] Finished. Writing result to shared memory.")
    #         shared_results[gpu_id] = (inds_list, seq_scores_list, ctrl_scores_list)

    #     except Exception as e:
    #         logging.error(f"[GPU {gpu_id}] Worker failed: {e}")
    #         logging.error(traceback.format_exc())
    #         shared_results[gpu_id] = ([], [], [])
    
    
    def _evaluate_worker(self, dataset_split, gpu_id, output_file, shared_results, progress_bar):
        try:
            torch.cuda.set_device(gpu_id)
            self.device = torch.device(f"cuda:{gpu_id}")
            self.model.to(self.device)

            dataloader = DataLoader(dataset_split, batch_size=self.batch_size, shuffle=False, num_workers=0)

            inds_list = []
            seq_scores_list = []
            ctrl_scores_list = []

            with open(output_file, "w") as f:
                f.write("idx\tseq_score\tctrl_score\n")
                pbar = tqdm(dataloader, disable=(not progress_bar), ncols=120, position=gpu_id, desc=f"[GPU {gpu_id}]")

                for batch_idx, (seq_tokens, ctrl_tokens, inds) in enumerate(pbar):
                    try:
                        seq_tokens = seq_tokens.to(self.device)
                        ctrl_tokens = ctrl_tokens.to(self.device)

                        seq_mask = (seq_tokens != self.tokenizer.pad_token_id).long()
                        ctrl_mask = (ctrl_tokens != self.tokenizer.pad_token_id).long()

                        seq_starts = torch.zeros(seq_tokens.size(0), dtype=torch.long, device=self.device)
                        ctrl_starts = torch.zeros(ctrl_tokens.size(0), dtype=torch.long, device=self.device)

                        seq_ends = seq_mask.sum(dim=1)
                        ctrl_ends = ctrl_mask.sum(dim=1)

                        seq_scores = self.score(seq_tokens, seq_starts, seq_ends, seq_mask)
                        ctrl_scores = self.score(ctrl_tokens, ctrl_starts, ctrl_ends, ctrl_mask)

                        for ind, s_score, c_score in zip(inds.tolist(), seq_scores.flatten(), ctrl_scores.flatten()):
                            inds_list.append(ind)
                            seq_scores_list.append(float(s_score))
                            ctrl_scores_list.append(float(c_score))
                            f.write(f"{ind}\t{s_score}\t{c_score}\n")
                        f.flush()
                    except Exception as batch_err:
                        logging.error(f"[GPU {gpu_id}] Error in batch {batch_idx}: {batch_err}")
                        logging.error(traceback.format_exc())

            logging.info(f"[GPU {gpu_id}] Finished. Writing result to shared memory.")
            shared_results[gpu_id] = (inds_list, seq_scores_list, ctrl_scores_list)

        except Exception as e:
            logging.error(f"[GPU {gpu_id}] Worker failed: {e}")
            logging.error(traceback.format_exc())
            shared_results[gpu_id] = ([], [], [])


class HFZeroShotEvaluator(ZeroShotPairedControlEvaluator, metaclass=ABCMeta):
    def __init__(self, tokenizer, model, dataset, batch_size, num_workers, device):
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(device)
        super().__init__(dataset, batch_size, num_workers, device)

    @property
    @abstractmethod
    def start_token(self):
        pass

    @property
    @abstractmethod
    def end_token(self):
        pass
    
    @property
    def mask_token(self):
        return self.tokenizer.mask_token_id

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer.batch_encode_plus(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if self.start_token is not None:
            starts = torch.where(tokens == self.start_token)[1] + 1 
        else:
            starts = 0
        if self.end_token is not None:
            ends = torch.where(tokens == self.end_token)[1]
        else:
            ends = attention_mask.sum(dim=1) 

        return tokens, starts, ends, attention_mask 

    def model_fwd(self, tokens_in, attention_mask, tokens_out):
        with torch.no_grad():
            torch_outs = self.model(
                tokens_in,
                attention_mask=attention_mask,
            )
            logits = torch_outs.logits.swapaxes(1, 2)
            lls = -F.cross_entropy(logits, tokens_out, reduction="none")
        return lls
    

class DNABERT2Evaluator(HFZeroShotEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        with NoModule("triton"):
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2
    
    def score(self, tokens, starts, ends, attention_mask):
        return MaskedZeroShotScore.score(self, tokens, starts, ends, attention_mask)

class GenaLMEvaluator(HFZeroShotEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2
    
    def score(self, tokens, starts, ends, attention_mask):
        return MaskedZeroShotScore.score(self, tokens, starts, ends, attention_mask)


class HDEvaluator(HFZeroShotEvaluator, CausalZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return None
    
    @property
    def end_token(self):
        return 1
    
    def model_fwd(self, tokens_in, attention_mask, tokens_out):
        with torch.no_grad():
            torch_outs = self.model(
                tokens_in,
            )
            logits = torch_outs.logits.swapaxes(1, 2)
            lls = torch.zeros(tokens_out.shape[:2], device=self.device)
            lls[:,1:] = -F.cross_entropy(logits[:,:,:-1], tokens_out[:,1:], reduction="none")
        return lls
    
    def score(self, tokens, starts, ends, attention_mask):
        return MaskedZeroShotScore.score(self, tokens, starts, ends, attention_mask)
    

class CaduceusEvaluator(HFZeroShotEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return None
    
    @property
    def end_token(self):
        return 1

    def model_fwd(self, tokens_in, attention_mask, tokens_out):
        with torch.no_grad():
            torch_outs = self.model(
                tokens_in
            )
            logits = torch_outs.logits.swapaxes(1, 2)
            lls = -F.cross_entropy(logits, tokens_out, reduction="none")
        return lls


class MistralEvaluator(HFZeroShotEvaluator, CausalZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2
    
    def model_fwd(self, tokens_in, attention_mask, tokens_out):
        with torch.no_grad():
            torch_outs = self.model(
                tokens_in,
                attention_mask=attention_mask,
            )
            logits = torch_outs.logits.swapaxes(1, 2)
            lls = torch.zeros(tokens_out.shape[:2], device=self.device)
            lls[:,1:] = -F.cross_entropy(logits[:,:,:-1], tokens_out[:,1:], reduction="none")
        return lls


class NTEvaluator(HFZeroShotEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 3
    
    @property
    def end_token(self):
        return None
    
    def score(self, tokens, starts, ends, attention_mask):
        return MaskedZeroShotScore.score(self, tokens, starts, ends, attention_mask)
    

class GENERatorEvaluator(HFZeroShotEvaluator, CausalZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return None
    
    @property
    def end_token(self):
        return 1
    
    def model_fwd(self, tokens_in, attention_mask, tokens_out):
        with torch.no_grad():
            torch_outs = self.model(
                tokens_in,
            )
            logits = torch_outs.logits.swapaxes(1, 2)
            lls = torch.zeros(tokens_out.shape[:2], device=self.device)
            lls[:,1:] = -F.cross_entropy(logits[:,:,:-1], tokens_out[:,1:], reduction="none")
        return lls
    
    def score(self, tokens, starts, ends, attention_mask):
        return MaskedZeroShotScore.score(self, tokens, starts, ends, attention_mask)