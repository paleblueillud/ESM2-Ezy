# -*-coding:utf-8-*-
import argparse
import contextlib
import os
import sys

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import LaccaseModel, resolve_dtype
from dataset import TrainingDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_positive_data', type=str)
    parser.add_argument('--train_negative_data', type=str)
    parser.add_argument('--test_positive_data', type=str)
    parser.add_argument('--test_negative_data', type=str)
    parser.add_argument('--model_path', type=str, default="esm2_t36_3B_UR50D")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--last_layers', type=int, default=1)
    parser.add_argument('--save_path', type=str, default=".")
    # Add Early Stop parameter with default=None to indicate disabled
    parser.add_argument('--patience', type=int, default=None, help='Number of epochs to wait before early stop. If not provided, early stop is disabled.')
    parser.add_argument('--dtype', type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument('--pos-weight', type=float, default=2.0, dest="pos_weight")
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--grad-accum-steps', type=int, default=8, dest="grad_accum_steps")
    parser.add_argument('--lr-backbone', type=float, default=1e-5, dest="lr_backbone")
    parser.add_argument('--lr-head', type=float, default=1e-4, dest="lr_head")
    parser.add_argument('--weight-decay', type=float, default=0.01, dest="weight_decay")
    parser.add_argument('--early-stop-metric', type=str, default="recall", choices=["recall", "f1", "accuracy"])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    train_positive_data = args.train_positive_data
    train_negative_data = args.train_negative_data
    test_positive_data = args.test_positive_data
    test_negative_data = args.test_negative_data
    model_path = args.model_path
    BATCH_SIZE = int(args.batch_size)
    EPOCH = int(args.epoch)
    last_layers = int(args.last_layers)
    total_save_path = args.save_path
    patience = args.patience  # None if not provided
    dtype_name = args.dtype
    pos_weight = float(args.pos_weight)
    threshold = float(args.threshold)
    grad_accum_steps = int(args.grad_accum_steps)
    lr_backbone = float(args.lr_backbone)
    lr_head = float(args.lr_head)
    weight_decay = float(args.weight_decay)
    early_stop_metric = args.early_stop_metric
    model_dtype = resolve_dtype(dtype_name)

    if grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    if device.type == "cpu" and model_dtype != torch.float32:
        print("Warning: non-float32 dtype on CPU is not supported reliably. Falling back to float32.")
        model_dtype = torch.float32
        dtype_name = "float32"
    use_amp = device.type == "cuda" and model_dtype != torch.float32
    scaler = torch.cuda.amp.GradScaler() if use_amp and model_dtype == torch.float16 else None

    def autocast_context():
        return torch.autocast(device_type="cuda", dtype=model_dtype) if use_amp else contextlib.nullcontext()
    
    # ==================== Early Stop Initialization ====================
    best_metric = -1.0
    best_acc = 0.0           # Record the best accuracy
    best_recall = 0.0
    best_f1 = 0.0
    best_epoch = -1          # Record the epoch where best accuracy occurred
    wait_counter = 0         # Counter for no improvement
    if patience is not None:
        print(f"Early stop enabled with patience={patience}")
    else:
        print("Early stop disabled")
    
    # model
    print("Loading model...")
    model = LaccaseModel(model_path, dtype=dtype_name)
    for name, param in model.named_parameters():
        param.requires_grad = False
        for last_layer in range(1, last_layers+1):
            if f"layers.{model.layers-last_layer}." in name:
                param.requires_grad = True
        if "dnn" in name:
            param.requires_grad = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    model = model.to(device=device, dtype=model_dtype)
    
    weight = torch.tensor([1.0, pos_weight], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("dnn"):
            head_params.append(param)
        else:
            backbone_params.append(param)
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]
    )
    # data
    train_dataset = TrainingDataset(positive_path=train_positive_data, negative_path=train_negative_data, dynamic_negative_sampling=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                    collate_fn=train_dataset.collate_fn, drop_last=False, pin_memory=True)
    test_dataset = TrainingDataset(positive_path=test_positive_data, negative_path=test_negative_data)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                    collate_fn=test_dataset.collate_fn, drop_last=False, pin_memory=True)

    Test_Acc = []
    ckpt_dir = os.path.join(total_save_path, "ckpt", f"dnn_model_lastlayer{last_layers}")
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(EPOCH):
        # ==================== Early Stop Check ====================
        if patience is not None and wait_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}!")
            print(f"Best test accuracy {best_acc:.4f} achieved at epoch {best_epoch}")
            break
        
        # train
        model.train()
        print(len(train_dataloader))
        optimizer.zero_grad()
        for i, item in enumerate(train_dataloader):
            content, label = item
            label = label.to(device)
            with autocast_context():
                last_result = model(content)
                loss = criterion(last_result, label) / grad_accum_steps
            print("epoch: {} \t iteration : {} \t Loss: {} \t lr: {}".format(epoch, i, loss.item(),
                                                                             optimizer.param_groups[0]['lr']), flush=True)
            # break
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_dataloader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            if i % (len(train_dataloader)//2) == 0:
                # eval
                model.eval()
                total_test = 0
                tp = fp = tn = fn = 0
                for m, test in enumerate(tqdm(test_dataloader)):
                    data_test, label_test = test
                    label_test = label_test.to(device)

                    with torch.no_grad():
                        with autocast_context():
                            last_result_test = model(data_test)

                    probs = torch.softmax(last_result_test, dim=1)[:, 1]
                    predicted = (probs >= threshold).long()
                    tp += ((predicted == 1) & (label_test == 1)).sum().item()
                    fp += ((predicted == 1) & (label_test == 0)).sum().item()
                    tn += ((predicted == 0) & (label_test == 0)).sum().item()
                    fn += ((predicted == 0) & (label_test == 1)).sum().item()
                    total_test += label_test.size(0)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                current_acc = (tp + tn) / total_test if total_test > 0 else 0.0
                Test_Acc.append(current_acc)
                print(
                    "Epoch_item: {} \t\t total: {} \t\t Accuracy: {:.6f} \t Precision: {:.6f} \t Recall: {:.6f} \t F1: {:.6f} \n".format(
                        epoch, total_test, current_acc, precision, recall, f1
                    )
                )
                result_dir = f"{total_save_path}/result"
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                with open(f"{result_dir}/dnn_result_test_lastlayer{last_layers}.txt", "a+", encoding="utf-8") as output:
                    output.write(
                            "Epoch_item: {} \t\t total: {} \t\t Accuracy: {:.6f} \t Precision: {:.6f} \t Recall: {:.6f} \t F1: {:.6f} \t Threshold: {:.3f}\n".format(
                                epoch, total_test, current_acc, precision, recall, f1, threshold))
                    output.write(f"TP {tp}\tFP {fp}\tTN {tn}\tFN {fn}\n")

                with open(f"{result_dir}/dnn_result_test_ACC_lastlayer{last_layers}.txt", "a+", encoding="utf-8") as output:
                    output.write(str(Test_Acc) + "\n")
                    
                metric_value = {
                    "accuracy": current_acc,
                    "recall": recall,
                    "f1": f1,
                }[early_stop_metric]
                improved = metric_value > best_metric
                if improved:
                    best_metric = metric_value
                    best_acc = current_acc
                    best_recall = recall
                    best_f1 = f1
                    best_epoch = epoch
                    wait_counter = 0
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
                    trainable_state = {
                        k: v.detach().cpu()
                        for k, v in model.state_dict().items()
                        if k in trainable_keys
                    }
                    torch.save(
                        {
                            "state_dict": trainable_state,
                            "meta": {
                                "base_model": model.base_model,
                                "last_layers": last_layers,
                                "repr_dim": model.repr_dim,
                                "dtype": dtype_name,
                            },
                        },
                        best_path,
                    )
                    print(
                        f"Test {early_stop_metric} improved to {metric_value:.4f} at epoch {best_epoch} "
                        f"(acc {current_acc:.4f}, recall {recall:.4f}, f1 {f1:.4f})"
                    )
                elif patience is not None:
                    wait_counter += 1
                    print(f"Test {early_stop_metric} not improved. Patience: {wait_counter}/{patience}")
                    
                model.train()
        
        # save model
        trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
        trainable_state = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if k in trainable_keys
        }
        torch.save(
            {
                "state_dict": trainable_state,
                "meta": {
                    "base_model": model.base_model,
                    "last_layers": last_layers,
                    "repr_dim": model.repr_dim,
                    "dtype": dtype_name,
                },
            },
            os.path.join(ckpt_dir, f"epoch{epoch}.pth"),
        )
        
    # ==================== Final Report ====================
    if patience is not None:
        print(
            f"\nTraining completed with early stop! Best {early_stop_metric}: {best_metric:.4f} "
            f"(acc {best_acc:.4f}, recall {best_recall:.4f}, f1 {best_f1:.4f}) at epoch {best_epoch}"
        )
    else:
        print(
            f"\nTraining completed without early stop. Best {early_stop_metric}: {best_metric:.4f} "
            f"(acc {best_acc:.4f}, recall {best_recall:.4f}, f1 {best_f1:.4f}) at epoch {best_epoch}"
        )
