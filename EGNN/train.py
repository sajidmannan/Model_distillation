import numpy as np
import random
import argparse
import torch 
import torch.nn as nn
from torch.autograd import grad 
import torch_geometric 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import global_add_pool
import sys 
import time
import wandb
from tqdm import tqdm
sys.path.append("egnn/")
from models.egnn_clean import egnn_clean as eg
from model import EGNN_Ener


def loss_fn(pred_forces, batch):
    if 'is_root' in batch:
        root_nodes = batch['is_root']
        return torch.mean(torch.square(batch["forces"][root_nodes] - pred_forces[root_nodes]))
    else:
        return torch.mean(torch.square(batch["forces"] - pred_forces))

def compute_rmse(delta):
    return np.sqrt(np.mean(np.square(delta))).item() * 1e3    # ev/A -> mev/A

def compute_mape(delta, target_val):
    return np.mean(np.abs(delta / (target_val + 1e-9))).item() * 100.0

def compute_mae(delta: np.ndarray) -> float:
    return np.mean(np.abs(delta)).item() * 1e3   # ev/A -> mev/A

def train_epoch(model, train_loader, device, optimizer):
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc="Training..."):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred_forces = model(batch, device)
        loss = loss_fn(pred_forces, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().detach().item()
    
    return train_loss / len(train_loader)

def evaluate(model, valid_loader, device):
    delta_fs_list = []
    fs_list = []
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation..."):
            batch = batch.to(device)
            pred_forces = model(batch, device)
            loss = loss_fn(pred_forces, batch)
            val_loss += loss.cpu().detach().item()
            delta_fs_list.append(batch["forces"] - pred_forces)
            fs_list.append(batch["forces"])

    delta_fs = torch.cat(delta_fs_list, dim=0).cpu().detach().numpy()
    fs = torch.cat(fs_list, dim=0).cpu().detach().numpy()

    val_rmse = compute_rmse(delta_fs)
    val_mape = compute_mape(delta_fs, fs)
    val_mae = compute_mae(delta_fs)
    val_loss /= len(valid_loader)

    return val_loss, val_rmse, val_mape, val_mae


def train(args):
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    train_dataset = torch.load(args.train_file)
    valid_dataset = torch.load(args.valid_file)
    if args.sample_frac is not None:
        random.seed(args.seed)
        n_samples = int(len(train_dataset) * args.sample_frac)
        train_dataset = random.sample(train_dataset, n_samples)
    # if args.clip_threshold is not None:
    #     train_dataset = [data for data in train_dataset if data["count"] >= args.clip_threshold]
    device = torch.device(args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False)

    model = EGNN_Ener(node_input_dim=train_dataset[0]['node_attrs'].shape[1],
                      node_output_dim=16,
                      edge_input_dim=3, 
                      hidden_dim=64,
                      num_layers=2
                      ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=50, min_lr=1e-7
            )
    
    wandb.login(key="e71ce0905b29dea3fb273c623d6f147b8de2db6b")
    wandb.init(
        # set the wandb project where this run will be logged
        project="EGNN (Satorras)",
        config=vars(args), 
        mode='online'
    )

    best_val_loss = float('inf')
    best_rmse = float('inf')
    best_mape = float('inf')
    best_mae = float('inf')
    best_epoch = 0

    training_start = time.perf_counter()
    
    for epoch in range(args.epochs):
        train_epoch_loss = train_epoch(model, train_loader, device, optimizer)
        val_loss, val_rmse, val_mape, val_mae = evaluate(model, valid_loader, device)

        print(f"Train Loss: {train_epoch_loss:.3f}, Val Loss: {val_loss:.3f}, RMSE (mev/A): {val_rmse:.3f}, MAPE: {val_mape:.3f}, MAE (meV/A): {val_mae:.3f}")
        wandb.log( {
                    'train_loss': train_epoch_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_mape': val_mape,
                    'val_mae': val_mae
                }, step = epoch)
    
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_rmse = val_rmse
            best_mape = val_mape
            best_mae = val_mae
            best_epoch = epoch + 1
            # if args.clip_threshold is None:
            #     torch.save(model.state_dict(), f"Models/{args.dataset}/{args.type}_{args.percent}p.model")
            # else:
            #     torch.save(model.state_dict(), f"Models/{args.dataset}/{args.type}_{args.percent}p_clip{args.clip_threshold}.model")
            # print(f"Epoch {epoch+1}: Best model saved with val loss {val_loss:.4f}")

    training_end = time.perf_counter()
    print(f"Training completed!")
    wandb.log({
        "Total training time (s)": training_end-training_start,
        "Best RMSE (ev/A)": best_rmse,
        "Best MAPE": best_mape,
        "Best MAE": best_mae,
        "Best Epoch": best_epoch,
    })


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Training EGNN")
    parser.add_argument('--dataset', type=str, required=True, help="System")
    parser.add_argument('--type', type=str, required=True, choices=['distil', 'full', 'random'], help="distil/full")
    parser.add_argument('--sample_frac', type=float, required=False, help="to be used during randomly sampling from full dataset")
    parser.add_argument('--seed', type=int, required=False, help="for random sampling")
    parser.add_argument('--percent', type=int, default=100, required=False, help="percentage of ego graphs compressed from")
    parser.add_argument('--clip_threshold', type=int, default=None, required=False, help="Tail clipping threshold")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--valid_batch_size', type=int, default=20, help="Batch size for validation")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument('--train_file', type=str, required=True, help="Training data path")
    parser.add_argument('--valid_file', type=str, required=True, help="Validation data path")
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda','cpu'], help="Device to train on")
    args = parser.parse_args()

    train(args)


