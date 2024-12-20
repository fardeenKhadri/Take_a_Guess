import torch
from model import LSTMModel

def test_model_structure():
    """
    Test if the model is correctly initialized and its forward pass works as expected.
    """
    try:
        model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
        print("Model structure initialized successfully.")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return False
    return True

def test_forward_pass():
    """
    Test the forward pass of the model with dummy input.
    """
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    dummy_input = torch.randn(10, 50, 1)  # Batch size 10, sequence length 50, feature size 1
    try:
        output = model(dummy_input)
        assert output.shape == (10, 1), f"Output shape mismatch: Expected (10, 1), got {output.shape}"
        print("Forward pass test passed.")
    except Exception as e:
        print(f"Forward pass test failed: {e}")
        return False
    return True

if __name__ == "__main__":
    print("Running tests for LSTM model...")
    structure_test = test_model_structure()
    forward_test = test_forward_pass()
    if structure_test and forward_test:
        print("All model tests passed successfully!")
    else:
        print("Some tests failed. Check the logs above.")
