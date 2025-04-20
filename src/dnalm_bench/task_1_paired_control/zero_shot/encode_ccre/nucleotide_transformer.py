import os
import argparse

from ..evaluators import PairedControlDataset, NTEvaluator
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot likelihood evaluation on cCREs.")
    parser.add_argument("--model_name", type=str, default="nucleotide-transformer-v2-500m-multi-species", help="Name of the model")
    parser.add_argument("--model_path", type=str, default="./Model/Genome/Nucleotide-Transformer/nucleotide-transformer-v2-500m-multi-species", help="Path to model checkpoint")
    parser.add_argument("--chroms", type=str, nargs="+", default=[
        "chr5", "chr10", "chr14", "chr18", "chr20", "chr22"],
        help="List of chromosomes to evaluate.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--work_dir", type=str, default=os.environ.get("DART_WORK_DIR", ""), help="Base working directory.")
    return parser.parse_args()


def main():
    args = parse_args()

    genome_fa = os.path.join(args.work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(args.work_dir, "task_1_ccre/processed_data/ENCFF420VPZ_processed.tsv")
    out_dir = os.path.join(args.work_dir, f"task_1_ccre/zero_shot_outputs/likelihoods/{args.model_name}")

    # dataset = PairedControlDataset(genome_fa, elements_tsv, args.chroms, args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = PairedControlDataset(
        genome_fa,
        elements_tsv,
        args.chroms,
        args.seed,
        tokenizer=tokenizer,
        cache_token=True
    )
    
    evaluator = NTEvaluator(args.model_path, dataset, args.batch_size, args.num_workers, args.device)
    metrics = evaluator.evaluate(out_dir, progress_bar=True)

    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()