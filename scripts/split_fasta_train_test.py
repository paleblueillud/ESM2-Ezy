#!/usr/bin/env python3
import argparse
import os
import random
from typing import List, Tuple, Optional

Record = Tuple[str, str]


def read_fasta(path: str) -> List[Record]:
    records: List[Record] = []
    header: Optional[str] = None
    seq_lines: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, "".join(seq_lines)))
    return records


def write_fasta(records: List[Record], path: str, wrap: int = 60) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for header, seq in records:
            handle.write(f"{header}\n")
            if wrap <= 0:
                handle.write(f"{seq}\n")
                continue
            for i in range(0, len(seq), wrap):
                handle.write(seq[i : i + wrap] + "\n")


def filter_by_length(records: List[Record], max_len: int) -> Tuple[List[Record], int]:
    kept: List[Record] = []
    removed = 0
    for header, seq in records:
        if len(seq) > max_len:
            removed += 1
            continue
        kept.append((header, seq))
    return kept, removed


def split_records(
    records: List[Record],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Record], List[Record]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    train_size = int(len(shuffled) * train_ratio)
    train = shuffled[:train_size]
    test = shuffled[train_size:]
    return train, test


def _check_ratio(name: str, r: float) -> None:
    if not (0.0 < r < 1.0):
        raise SystemExit(f"{name} must be between 0 and 1 (exclusive). Got {r!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split positive/negative FASTA files into train/test sets with length filtering."
    )
    parser.add_argument(
        "--positive",
        default="pazy_all_nsp_protein_names.fasta",
        help="Path to positive FASTA file.",
    )
    parser.add_argument(
        "--negative",
        default="uniprotkb_NOT_ec_3_1_1_101_NOT_ec_3_1_1_2026_01_15.fasta",
        help="Path to negative FASTA file.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1022,
        help="Maximum sequence length to keep.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fallback fraction of sequences to put in the training set (used if --pos-train-ratio/--neg-train-ratio are not set).",
    )
    parser.add_argument(
        "--pos-train-ratio",
        type=float,
        default=None,
        help="Fraction of POSITIVE sequences to put in the training set. Overrides --train-ratio for positives.",
    )
    parser.add_argument(
        "--neg-train-ratio",
        type=float,
        default=None,
        help="Fraction of NEGATIVE sequences to put in the training set. Overrides --train-ratio for negatives.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible splits.",
    )
    parser.add_argument(
        "--same-seed-for-both",
        action="store_true",
        help="If set, positives and negatives use the exact same seed (otherwise negatives use seed+1).",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write output FASTA files.",
    )
    parser.add_argument(
        "--pos-name",
        default="positive",
        help="Basename for positive outputs (e.g., positive_train.fasta).",
    )
    parser.add_argument(
        "--neg-name",
        default="negative",
        help="Basename for negative outputs (e.g., negative_train.fasta).",
    )
    parser.add_argument(
        "--wrap",
        type=int,
        default=60,
        help="Line wrap length for FASTA sequences. Use 0 for no wrapping.",
    )

    args = parser.parse_args()

    pos_ratio = args.train_ratio if args.pos_train_ratio is None else args.pos_train_ratio
    neg_ratio = args.train_ratio if args.neg_train_ratio is None else args.neg_train_ratio

    _check_ratio("--train-ratio", args.train_ratio)
    _check_ratio("--pos-train-ratio (resolved)", pos_ratio)
    _check_ratio("--neg-train-ratio (resolved)", neg_ratio)

    pos_records = read_fasta(args.positive)
    neg_records = read_fasta(args.negative)

    pos_records, pos_removed = filter_by_length(pos_records, args.max_len)
    neg_records, neg_removed = filter_by_length(neg_records, args.max_len)

    pos_seed = args.seed
    neg_seed = args.seed if args.same_seed_for_both else args.seed + 1

    pos_train, pos_test = split_records(pos_records, pos_ratio, pos_seed)
    neg_train, neg_test = split_records(neg_records, neg_ratio, neg_seed)

    os.makedirs(args.out_dir, exist_ok=True)

    pos_train_path = os.path.join(args.out_dir, f"{args.pos_name}_train.fasta")
    pos_test_path = os.path.join(args.out_dir, f"{args.pos_name}_test.fasta")
    neg_train_path = os.path.join(args.out_dir, f"{args.neg_name}_train.fasta")
    neg_test_path = os.path.join(args.out_dir, f"{args.neg_name}_test.fasta")

    write_fasta(pos_train, pos_train_path, wrap=args.wrap)
    write_fasta(pos_test, pos_test_path, wrap=args.wrap)
    write_fasta(neg_train, neg_train_path, wrap=args.wrap)
    write_fasta(neg_test, neg_test_path, wrap=args.wrap)

    print(f"Positive total (filtered): {len(pos_records)}; removed > {args.max_len}: {pos_removed}")
    print(f"Negative total (filtered): {len(neg_records)}; removed > {args.max_len}: {neg_removed}")
    print(f"Positive train {len(pos_train)} (ratio {pos_ratio}), test {len(pos_test)}")
    print(f"Negative train {len(neg_train)} (ratio {neg_ratio}), test {len(neg_test)}")
    print("Outputs:")
    print(pos_train_path)
    print(pos_test_path)
    print(neg_train_path)
    print(neg_test_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
