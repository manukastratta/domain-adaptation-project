from wilds import get_dataset

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="camelyon17", download=True, unlabeled=True)