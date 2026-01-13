import torch
import numpy as np
import mlflow
import mlflow.pytorch

from model import LSTMAutoEncoder

X = torch.tensor(
    np.load("data/processed/train_normal.npy"),
    dtype=torch.float32
)

loader = torch.utils.data.DataLoader(
    X, 
    batch_size=32,
    shuffle=True
)

model = LSTMAutoEncoder(n_features = X.shape[2])
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
criterion = torch.nn.MSELoss()

mlflow.set_experiment("Battery_LSTM_anomaly_Detection")

print("!!! Starting Training !!!")

with mlflow.start_run():
    NUM_EPOCHS = 50

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            remoc = model(batch)
            loss = criterion(remoc, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() 

        avg_loss = total_loss / len(loader)
        mlflow.log_metric("avg_loss", avg_loss, step=epoch)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

        torch.save(model.state_dict(), "models/lstm_model_v1.pt")
        mlflow.pytorch.log_model(model, "lstm_autoencoder")

print("Done Training!!!")


