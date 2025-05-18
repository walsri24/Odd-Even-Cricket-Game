import torch
import torch.nn as nn
import torch.optim as optim
from ai_model import LSTMModel
from utils import load_sequences

def prepare_data(sequences, seq_len=5):
    X, y = [], []
    for seq in sequences:
        if len(seq) > seq_len:
            for i in range(len(seq) - seq_len):
                X.append(seq[i:i+seq_len])
                y.append(seq[i+seq_len] - 1)
    return torch.tensor(X), torch.tensor(y)

def train():
    sequences = load_sequences()
    X, y = prepare_data(sequences)
    model = LSTMModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "data/model.pth")
    print("Model saved to data/model.pth")

if __name__ == "__main__":
    train()
