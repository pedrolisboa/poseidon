import torch
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(train_dataset_fold, device):
    """
    Calculate class weights for the training dataset to handle class imbalance.
    
    Args:
        train_dataset_fold (TimeSeriesPairDataset): The training dataset.
        
    Returns:
        np.ndarray: Class weights for each class.
    """
    _, _, labels = zip(*train_dataset_fold)
    labels = torch.tensor(labels, device=device)
    labels = np.array([label.cpu().numpy() for label in labels])
    
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return class_weights