from transformers import EEGDenoisingConfig, EEGDenoisingModel
from torch.utils.data import DataLoader
from utils import TUARDataset

# Initialize
config = EEGDenoisingConfig(num_channels=64)
model = EEGDenoisingModel(config)
dataset = TUARDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        denoised = model(batch)
        loss = torch.nn.functional.mse_loss(denoised, batch)  # Replace with clean data
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")