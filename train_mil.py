import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import DataParallel
from mil.varmil import VarMIL
from ovca import OVCA, OVMIL

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


def train_varmil(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        out, A = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = out.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return running_loss / (batch_idx + 1), correct / total

def val_mil(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            out, A = model(data)
            loss = criterion(out, target)

            running_loss += loss.item()
            _, predicted = out.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return running_loss / (batch_idx + 1), correct / total

def create_splits(dataset, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    val_size = dataset_size - train_size - test_size
    return random_split(dataset, [train_size, test_size, val_size])

def main():
    # Define parameters and instantiate required objects
    parser = argparse.ArgumentParser(description='Trains a VarMIL Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='', 
                        help='root_dir of image embeddings as .pt files. Path structure: root_dir/class_label/*.pt')
    parser.add_argument('--histotypes', action='store', type=str, nargs="+", 
                        help='space separated str IDs specifying histotype labels')
    parser.add_argument('--save', type=str, default='', 
                        help='path to save the trained model. will be created if it does not exist. will be saved as model_args-run_epoch.pt')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--run', type=str, default='0')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--lr', '-m', type=float, default=0.001)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.01)
    parser.add_argument('--ngpu', type=int, default=1)

    print("Parsing arguments...")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join(args.save, args.run), exist_ok=True)

    print("Loading data...")
    # Load the dataset
    train_set = OVMIL(dataset=args.dataset, split=0, histotypes=args.histotypes)
    val_set = OVMIL(dataset=args.dataset, split=1, histotypes=args.histotypes)
    test_set = OVMIL(dataset=args.dataset, split=2, histotypes=args.histotypes)
    if args.debug:
        print(f"train_set: {len(train_set)}, train_set[0][0] emb tensor.shape: {train_set[0][0].shape}, train_set[0][1] label: {train_set[0][1]}")
        print(f"train_set[0][0][:5]: {train_set[0][0][:5]}")
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Creating model...")
    # Define variables for training
    in_size = train_set[0][0].shape[1] # 1024
    model = VarMIL(in_size)
    model.to(device)
    if args.ngpu > 1:
        model = DataParallel(model, device_ids=list(range(args.ngpu)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    print("Training...")
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_varmil(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_acc = val_mil(model, val_dataloader, criterion, device)
        print('Epoch {0:3d} | Train Loss {1:.4f} | Train Acc {2:.3f} | Val Loss {3:.3f} | Val Acc {4:.4f}'.format(
                epoch, train_loss, train_acc, val_loss, val_acc))

        if epoch % args.save_freq == 0:
            save_path = os.path.join(args.save, f'model_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

    test_loss, test_acc = val_mil(model, test_dataloader, criterion, device)
    print(f"\nTesting, Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # Save the trained model
    save_path = os.path.join(args.save, f'model_end.pth')
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()