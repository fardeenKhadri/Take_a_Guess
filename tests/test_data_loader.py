from data_loader import load_and_preprocess_data

def test_data_loader():
    train_dataset, val_dataset, test_dataset, scaler = load_and_preprocess_data('../data/raw/indexProcessed.csv')

    assert len(train_dataset) > 0, "Training dataset is empty"
    assert len(val_dataset) > 0, "Validation dataset is empty"
    assert len(test_dataset) > 0, "Test dataset is empty"
    assert scaler is not None, "Scaler is None"
    
    print("Data loader test passed!")

if __name__ == "__main__":
    test_data_loader()
