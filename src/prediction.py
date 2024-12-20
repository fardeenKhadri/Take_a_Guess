import torch
from model import LSTMModel
from data_loader import load_and_preprocess_data

def predict(data_path, sequence_length=50):
    _, _, _, scaler = load_and_preprocess_data(data_path, sequence_length=sequence_length)

    model = LSTMModel()
    model.load_state_dict(torch.load("models/saved/stock_lstm_model.pth"))
    model.eval()

    data = scaler.data_max_
    input_seq = torch.tensor(data[-sequence_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        prediction = model(input_seq).item()
    
    predicted_price = scaler.inverse_transform([[prediction]])
    print(f"Predicted Price: {predicted_price[0][0]}")
