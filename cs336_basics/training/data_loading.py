import numpy as np
import torch

def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    # Sample random starting positions
    # Valid range: [0, len(dataset) - context_length - 1]
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    
    # Build input and target tensors
    x = torch.tensor([dataset[i : i + context_length] for i in starts], dtype=torch.long)
    y = torch.tensor([dataset[i + 1 : i + context_length + 1] for i in starts], dtype=torch.long)
    
    return x.to(device), y.to(device)