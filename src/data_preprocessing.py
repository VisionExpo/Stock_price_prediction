import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    training_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[0:training_size, :]
    test_data = scaled_data[training_size:len(scaled_data), :1]
    
    return train_data, test_data, scaler

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):  # Removed the -1 to include all samples
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

if __name__ == "__main__":
    # Example usage
    pass
