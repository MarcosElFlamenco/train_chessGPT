# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
import pickle
import argparse
import pandas as pd

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
dtype = np.uint8  # Currently there are only 32 tokens in the chess LLMs vocab

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse PGN files and filter by ELO range and transcript length")
    parser.add_argument('--blocks_file', type=str, required=True, help='Directory containing PGN files')
    parser.add_argument('--bin_dir', type=str, required=True, help='Directory containing PGN files')
    parser.add_argument('--test_size', type=float, required=True, help='Directory containing PGN files')
    args = parser.parse_args()

    file_path = args.blocks_file
    bin_dir = args.bin_dir
    test_size = args.test_size

    try:
        dataset = load_dataset('csv', data_files=file_path)
        print(dataset)

    except Exception as e:
        print(f"Error loading dataset: {e}")

    nec_test_size = 1024 / len(dataset['train'])
    test_size = max(test_size, nec_test_size)
    # by default only contains the 'train' split, so create a test split
    split_dataset = dataset['train'].train_test_split(
        test_size=test_size, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. Using meta.pkl in the same directory as this file
    meta_path = os.path.join(os.path.dirname(__file__), "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    itos = meta["itos"]

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint8, mode='r')
    # print(split_dataset["val"][0])
    # print(len(split_dataset["val"]["transcript"][0]))

    # For verifying that all games are 1024 tokens long
    # for game in split_dataset["train"]["transcript"]:
    #     if len(game) != 1024:
    #         print(len(game))
    #         print(game)
    #         break
    # print(stoi)

    column_name = "transcript"

    def process(example):
        ids = np.array([stoi[c] for c in example[column_name]], dtype=dtype)
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=[column_name],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # print(tokenized["val"]["ids"])


    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {arr_len} tokens")
        
        # Construct the file path
        try:
            filename = os.path.join(os.path.dirname(__file__), bin_dir, f"{split}.bin")
            
            # Ensure that the bin_dir exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Try to open the file with memmap
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            print(f"File {filename} created successfully or already exists with shape {arr.shape}")
            
        except FileNotFoundError as e:
            print(f"Error: File {filename} could not be created. {str(e)}")
            continue  # Skip to the next iteration if file creation fails

        total_batches = 1024
        idx = 0

        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")

            # Concatenate the batch of IDs
            arr_batch = np.concatenate(batch["ids"])

            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        
        # Flush data to disk
        arr.flush()