import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import contextlib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LaccaseModel, resolve_dtype
from dataset import FastaDataset

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="esm2_t36_3B_UR50D")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--inference_data', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dtype', type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parse_args()
    model_path = args.model_path
    checkpoint_path = args.checkpoint_path
    inference_data = args.inference_data
    output_path = args.output_path
    batch_size = args.batch_size
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

    # data
    print("Reading candidate data...")
    inference_dataset = FastaDataset(inference_data)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=inference_dataset.collate_fn, drop_last=False, pin_memory=True)
    
    inference_list = []
    with torch.no_grad():
        for content in tqdm(inference_dataloader, total=len(inference_dataloader)):
            with autocast_context():
                last_result = model(content)
            pred = torch.argmax(last_result, dim=1)
            mask = (pred == 1).cpu().tolist()
            inference_list.extend([c for m, c in zip(mask, content) if m])

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "candidate.fa"), "w") as f:
        for c in inference_list:
            f.write(f">{c[0]}\n{c[1]}\n")
