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
    model_dtype = resolve_dtype(dtype_name)

    if device.type == "cpu" and model_dtype != torch.float32:
        print("Warning: non-float32 dtype on CPU is not supported reliably. Falling back to float32.")
        model_dtype = torch.float32
        dtype_name = "float32"
    use_amp = device.type == "cuda" and model_dtype != torch.float32
    scaler = torch.cuda.amp.GradScaler() if use_amp and model_dtype == torch.float16 else None

    def autocast_context():
        return torch.autocast(device_type="cuda", dtype=model_dtype) if use_amp else contextlib.nullcontext()
    
    # ==================== Early Stop Initialization ====================
    best_acc = 0.0           # Record the best accuracy
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
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
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
        for i, item in enumerate(train_dataloader):
            content, label = item
            label = label.to(device)
            with autocast_context():
                last_result = model(content)
                loss = criterion(last_result, label)
            print("epoch: {} \t iteration : {} \t Loss: {} \t lr: {}".format(epoch, i, loss.item(),
                                                                             optimizer.param_groups[0]['lr']), flush=True)
            # break
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if i % (len(train_dataloader)//2) == 0:
                # eval
                model.eval()
                total_test = 0
                correct_test = 0
                predict_test = {}
                predict_really_test = {}
                ground_truth_test = {0:len(test_dataset.negative_dataset),1:len(test_dataset.positive_dataset)}
                for m, test in enumerate(tqdm(test_dataloader)):
                    data_test, label_test = test
                    label_test = label_test.to(device)

                    with torch.no_grad():
                        with autocast_context():
                            last_result_test = model(data_test)

                    # label
                    predicted = torch.argmax(last_result_test.data, dim=1)

                    predict_label = predicted.cpu().numpy()
                    really_label = label_test.cpu().numpy()

                    for k in range(len(predict_label)):
                        if predict_label[k] not in predict_test:
                            predict_test[predict_label[k]] = 1
                        else:
                            predict_test[predict_label[k]] += 1

                        if predict_label[k] == really_label[k]:
                            if predict_label[k] not in predict_really_test:
                                predict_really_test[predict_label[k]] = 1
                            else:
                                predict_really_test[predict_label[k]] += 1

                    total_test += label_test.size(0)
                    correct_test += (predicted == label_test).sum().cpu().item()


                out = ""
                for m in range(len(ground_truth_test)):
                    if m in predict_test and m in predict_really_test:
                        out = out + "Category_" + str(m) + "\t" + "predict_really " + str(predict_really_test[m]) + \
                                "\t" + "predict " + str(predict_test[m]) + "\t" + "     Precision " + str(
                                predict_really_test[m] / predict_test[m]) + "\t" \
                                + "recall" + str(predict_really_test[m] / ground_truth_test[m])[:5] + "\n"
                
                current_acc = correct_test / total_test
                Test_Acc.append(current_acc)
                print("Epoch_item: {} \t\t Correct_num: {} \t\t total: {} \t\t Accuracy on test data: {} \n".format(
                    epoch, correct_test, total_test, current_acc))
                print(out)
                result_dir = f"{total_save_path}/result"
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                with open(f"{result_dir}/dnn_result_test_lastlayer{last_layers}.txt", "a+", encoding="utf-8") as output:
                    output.write(
                            "Epoch_item: {} \t\t Correct_num: {} \t\t total: {} \t\t Accuracy on test data: {} \n".format(
                                epoch, correct_test, total_test, current_acc))
                    output.write(out)

                with open(f"{result_dir}/dnn_result_test_ACC_lastlayer{last_layers}.txt", "a+", encoding="utf-8") as output:
                    output.write(str(Test_Acc) + "\n")
                    
                improved = current_acc > best_acc
                if improved:
                    best_acc = current_acc
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
                    print(f"Test accuracy improved to {best_acc:.4f} at epoch {best_epoch}")
                elif patience is not None:
                    wait_counter += 1
                    print(f"Test accuracy not improved. Patience: {wait_counter}/{patience}")
                    
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
        print(f"\nTraining completed with early stop! Best test accuracy: {best_acc:.4f} at epoch {best_epoch}")
    else:
        print("\nTraining completed without early stop")
