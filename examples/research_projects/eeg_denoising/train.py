from transformers import EEGDenoisingModel, EEGDenoisingConfig
from transformers.models.eeg_denoising import EEGDenoisingProcessor
import torch
from torch.utils.data import Dataset, DataLoader

# Load TUAR data (simplified)
class TUARDataset(Dataset):
    def __init__(self, processor):
        self.processor = processor
        self.data = [...]  # Load TUAR EDF files here

    def __getitem__(self, idx):
        noisy = self.processor.process(self.data[idx])
        clean = ...  # Get clean reference (e.g., artifact-free segments)
        return torch.tensor(noisy), torch.tensor(clean)

# Initialize
config = EEGDenoisingConfig(num_channels=64, window_size=256)
model = EEGDenoisingModel(config)
processor = EEGDenoisingProcessor()

# Train
dataset = TUARDataset(processor)
dataloader = DataLoader(dataset, batch_size=32)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for noisy, clean in dataloader:
        optimizer.zero_grad()
        denoised = model(noisy)
        loss = torch.nn.functional.mse_loss(denoised, clean)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")