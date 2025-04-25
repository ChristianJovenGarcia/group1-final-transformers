import glob
import mne
from torch.utils.data import Dataset

class TUARDataset(Dataset):
    def __init__(self, data_dir="data/tuh_eeg/"):
        self.file_paths = glob.glob(f"{data_dir}/*.edf")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        raw = mne.io.read_raw_edf(self.file_paths[idx], preload=True)
        data = raw.get_data()  # Shape (channels, time)
        return torch.tensor(data, dtype=torch.float32)