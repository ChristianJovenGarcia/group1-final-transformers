from utils import TUARDataset

# Initialize the dataset
dataset = TUARDataset(fixed_length=20000)

# Check the dataset size
print(f"Dataset size: {len(dataset)}")

# Load the first EEG file
first_sample, past_time_features, past_observed_mask = dataset[0]
print(f"First sample shape: {first_sample.shape}")
print(f"Past time features shape: {past_time_features.shape}")
print(f"Past observed mask shape: {past_observed_mask.shape}")