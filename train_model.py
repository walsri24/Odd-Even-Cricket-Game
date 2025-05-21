from calendar import EPOCH

import torch
import torch.nn as nn
import torch.optim as optim
from ai_model import LSTMModel
from utils import load_sequences
import os

NUM_OF_EPOCHS = 50

def prepare_data(sequences, seq_len=5):
    X, y = [], []
    for seq in sequences:
        if len(seq) > seq_len:
            for i in range(len(seq) - seq_len):
                X.append(seq[i:i+seq_len])
                y.append(seq[i+seq_len] - 1)
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train():
    sequences = load_sequences()
    if not sequences:
        print("No sequences found – play a few Hard/Medium games first!")
        return

    print(f"Loaded {len(sequences)} sequences")
    X, y = prepare_data(sequences)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    X, y = X.to(device), y.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=64,
        shuffle=True
    )

    for epoch in range(NUM_OF_EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    os.makedirs("data", exist_ok=True)
    torch.save(model.state_dict(), "data/model.pth")
    print("✅ Model saved to data/model.pth")

if __name__ == "__main__":
    train()
