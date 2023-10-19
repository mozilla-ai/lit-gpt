"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import os
import sys
from pathlib import Path
from typing import List,Optional

import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer


def prepare(
    destination_path: Path = Path("data/pubmed-qa-instruct"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    mask_inputs: bool = False,
    data_file_name: str = "pubmed_qa_instruction_cleaned.json",
    data_repo_id: str = "FedML/PubMedQA_instruction",
    ignore_index: int = -1,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare the pubmedQA Instruction dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    if access_token is None:
        raise ValueError(
            "pubmedQA Instruction requires authentication, please set the `HF_TOKEN=your_token` environment"
            " variable or pass --access_token=your_token. You can find your token by visiting"
            " https://huggingface.co/settings/tokens"
        )
    
    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    data_file_path = destination_path / data_file_name
    print("Loading data file...")

    from datasets import load_dataset

    dataset = load_dataset(data_repo_id, use_auth_token=access_token)
    train_set = dataset["train"]
    test_set = dataset["test"]
    
    # download_if_missing(data_file_path, data_file_url)

    # with open(data_file_path, "r", encoding="utf-8") as file:
    #     data = file.readlines()
    #     data = [json.loads(line) for line in data]
    # for item in data:
    #     item["input"] = item.pop("context")
    #     item["output"] = item.pop("response")

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")

def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool, ignore_index: int) -> None:
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["response"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["context"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['context']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
