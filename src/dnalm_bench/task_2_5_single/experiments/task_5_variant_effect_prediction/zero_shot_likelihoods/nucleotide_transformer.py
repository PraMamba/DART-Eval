import os
import argparse
import polars as pl

from ....evaluators import NTVariantSingleTokenEvaluator
from ....components import VariantDataset

root_output_dir = os.environ.get("DART_WORK_DIR", "")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate variant effects using Nucleotide Transformer")
    parser.add_argument("--elements_tsv", type=str, required=True, help="Path to input variants BED file")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix name for output file")
    parser.add_argument("--genome_fa", type=str, required=True, help="Path to reference genome FASTA")
    parser.add_argument("--model_name", type=str, default="nucleotide-transformer-v2-500m-multi-species", help="Name of the model")
    parser.add_argument("--model_path", type=str, default="/data1/Mamba/Model/Genome/Nucleotide-Transformer/nucleotide-transformer-v2-500m-multi-species", help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cell_line", type=str, default="GM12878", help="Cell line name")
    return parser.parse_args()

def main():
    args = parse_args()

    out_dir = os.path.join(root_output_dir, f"task_5_variant_effect_prediction/outputs/zero_shot/likelihoods/{args.model_name}_1")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.output_prefix}.tsv")

    dataset = VariantDataset(args.genome_fa, args.elements_tsv, chroms=None, seed=args.seed)

    evaluator = NTVariantSingleTokenEvaluator(
        args.model_path,
        args.batch_size,
        args.num_workers,
        args.device
    )

    score_df = evaluator.evaluate(dataset=dataset, output_file=out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, score_df], how="horizontal")
    print(out_path)
    scored_df.write_csv(out_path, separator="\t")

if __name__ == "__main__":
    main()