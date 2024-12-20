import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import LSTMModel
from data_loader import load_and_preprocess_data

def train_model(data_path, sequence_length=50, epochs=20, batch_size=64, learning_rate=0.001):
    train_dataset, val_dataset, _, _ = load_and_preprocess_data(data_path, sequence_length=sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LSTMModel()
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.unsqueeze(-1), y.unsqueeze(-1)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.unsqueeze(-1), y.unsqueeze(-1)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")
    torch.save(model.state_dict(), "models/saved/stock_lstm_model.pth")
