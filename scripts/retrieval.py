import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import contextlib
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import faiss

from model import LaccaseModel, resolve_dtype
from dataset import FastaDataset

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="esm2_t36_3B_UR50D")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--candidate_data', type=str)
    parser.add_argument('--seed_data', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--candidate_batch_size', type=int, default=1)
    parser.add_argument('--seed_batch_size', type=int, default=1)
    parser.add_argument('--dtype', type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parse_args()
    model_path = args.model_path
    checkpoint_path = args.checkpoint_path
    candidate_data = args.candidate_data
    seed_data = args.seed_data
    output_path = args.output_path
    candidate_batch_size = args.candidate_batch_size
    seed_batch_size = args.seed_batch_size
    dtype_name = args.dtype
    model_dtype = resolve_dtype(dtype_name)

    if device.type == "cpu" and model_dtype != torch.float32:
        print("Warning: non-float32 dtype on CPU is not supported reliably. Falling back to float32.")
        model_dtype = torch.float32
        dtype_name = "float32"
    use_amp = device.type == "cuda" and model_dtype != torch.float32

    def autocast_context():
        return torch.autocast(device_type="cuda", dtype=model_dtype) if use_amp else contextlib.nullcontext()
    
    # load model
    print("Loading model...")
    model = LaccaseModel.from_pretrained(model_path, state_dict_path=checkpoint_path, dtype=dtype_name)
    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    print(model.device)

    # data
    print("Reading candidate data...")
    candidate_dataset = FastaDataset(candidate_data)
    candidate_dataloader = DataLoader(candidate_dataset, batch_size=candidate_batch_size, shuffle=False,
                                    collate_fn=candidate_dataset.collate_fn, drop_last=False, pin_memory=True)
    
    seed_dataset = FastaDataset(seed_data)
    seed_dataloader = DataLoader(seed_dataset, batch_size=seed_batch_size, shuffle=False,
                                    collate_fn=seed_dataset.collate_fn, drop_last=False, pin_memory=True)
    
    candidate_info_list = []
    with torch.no_grad():
        for j, item in tqdm(enumerate(candidate_dataloader), total=len(candidate_dataloader)):
            with autocast_context():
                out_result, last_repr = model(item, return_repr=True)
            last_repr_cpu = last_repr.detach().float().cpu().numpy()
            for i, r in zip(item, last_repr_cpu):
                candidate_info_list.append((i, r))
    candidate_repr = np.stack([r for item, r in candidate_info_list], axis=0)
    print(candidate_repr.shape)


    # faiss index
    index = faiss.IndexFlatL2(candidate_repr.shape[1])
    index.add(candidate_repr)
    result_list = []
    with torch.no_grad():
        for j, item in tqdm(enumerate(seed_dataloader), total=len(seed_dataloader)):
            with autocast_context():
                out_result, last_repr = model(item, return_repr=True)
            last_repr_cpu = last_repr.detach().float().cpu().numpy()
            D, I = index.search(last_repr_cpu, k=10)
            for i, distance in zip(I[0], D[0]):
                res_tuple = (item, candidate_info_list[i], distance)
                if res_tuple not in result_list:
                    result_list.append(res_tuple)

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "results.csv"), "w") as f:
        f.write("seed_id,candidate_id,candidate_sequence,distance\n")
        for res in result_list:
            seed_info, candidate_info, distance = res
            f.write(f"{seed_info[0][0]},{candidate_info[0][0]},{candidate_info[0][1]},{distance}\n")
