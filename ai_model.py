import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=6):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # predict next

def load_model(model_path="data/model.pth"):
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
