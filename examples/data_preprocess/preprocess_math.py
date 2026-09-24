"""Download QA-code and convert its math questions for the IGRPO environment."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests


DEFAULT_SOURCE = "https://raw.githubusercontent.com/RUC-NLPIR/ARPO/main/ARPO/rl_datasets"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "math"
MULTIPLE_CHOICE_MARKERS = (
    ("(A)", "(B)"), ("\nA)", "\nB)"), ("\nA.", "\nB."),
    (r"\textbf{(A)}", r"\textbf{(B)}"), ("\nA:", "\nB:"),
    ("- A)", "- B)"), ("- **A)**", "- **B)**"),
    (" A. ", " B. "), ("A. ", "B. "), ("**A.**", "**B.**"),
)


def download(session, url, destination):
    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                output.write(chunk)


def question_from_prompt(prompt):
    if len(prompt) == 2 and prompt[1]["role"] == "user":
        return prompt[1]["content"].strip()
    if len(prompt) == 1 and prompt[0]["role"] == "system":
        _, marker, question = prompt[0]["content"].partition("\nuser\n")
        if marker:
            return question.strip()
    raise ValueError("Expected QA-code system and user messages")


def convert_row(row, split, index, exclude_multiple_choice=False):
    question = question_from_prompt(row["prompt"])
    if not question:
        raise ValueError(f"{split} row {index}: empty question")
    if exclude_multiple_choice and any(
        first in question and second in question for first, second in MULTIPLE_CHOICE_MARKERS
    ):
        return None
    ground_truth = row["reward_model"]["ground_truth"]
    target = ground_truth.get("target") if isinstance(ground_truth, dict) else ground_truth
    if not isinstance(target, str):
        raise ValueError(f"{split} row {index}: missing math target")
    if not target.strip():
        return None
    ground_truth = {"target": target}
    return {
        "data_source": "math",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"index": index, "question": question, "split": split},
        "env_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": "math"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-base-url", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exclude-multiple-choice", action="store_true",
        help="Remove multiple choice questions to match TreeHCA's filtered math split.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir) as temporary, requests.Session() as session:
        staged = []
        for split, source in (("train", "train_10k.parquet"), ("val", "valid.parquet")):
            downloaded = Path(temporary) / source
            download(session, f"{args.source_base_url.rstrip('/')}/{source}", downloaded)
            frame = pd.read_parquet(downloaded)
            frame = frame[frame["ability"] == "math"]
            converted = [
                convert_row(row, split, i, args.exclude_multiple_choice)
                for i, row in frame.iterrows()
            ]
            converted = [row for row in converted if row is not None]
            staged_file = Path(temporary) / f"{split}.converted.parquet"
            pd.DataFrame(converted).to_parquet(staged_file, index=False)
            staged.append((staged_file, args.output_dir / f"{split}.parquet", len(converted)))
        for staged_file, output, count in staged:
            staged_file.replace(output)
            print(f"{output}: {count} math examples")


if __name__ == "__main__":
    main()
