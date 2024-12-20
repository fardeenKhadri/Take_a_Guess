import torch
from torch.utils.data import DataLoader
from model import LSTMModel
from data_loader import load_and_preprocess_data

def evaluate_model(data_path, sequence_length=50, batch_size=64):
    _, _, test_dataset, _ = load_and_preprocess_data(data_path, sequence_length=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = LSTMModel()
    model.load_state_dict(torch.load("models/saved/stock_lstm_model.pth"))
    model.eval()

    criterion = torch.nn.MSELoss()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.unsqueeze(-1), y.unsqueeze(-1)
            output = model(x)
            loss = criterion(output, y)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_loader)}")
