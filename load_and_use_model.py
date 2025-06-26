import os
import torch
import numpy as np
from tqdm import tqdm

# Import the data loader and model components
from train import load_model
from data_loader import RoundaboutTrajectoryDataLoader
from train import create_agent_mask

def load_and_test_model(model_path, data_path, batch_size=4):
    """
    Load a saved model and test it on sample data
    
    Args:
        model_path: Path to the saved model file
        data_path: Path to the test data file
        batch_size: Batch size for testing
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {model_path}")
    model, model_config, checkpoint = load_model(model_path, device)
    model.eval()
    
    # Print model information
    print("\nModel configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Print training metrics from the checkpoint
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if isinstance(metrics, dict) and 'val' in metrics:
            val_metrics = metrics['val']
            print("\nValidation metrics from checkpoint:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value}")
    
    # Create test dataset
    print(f"\nLoading test data from {data_path}")
    test_dataset = RoundaboutTrajectoryDataLoader(
        csv_path=data_path,
        obs_len=model_config.get('obs_len', 10),
        pred_len=model_config.get('pred_len', 10),
        dist_threshold=10.0,
        standardize_xy=True
    )
    
    test_loader = test_dataset.get_loader(batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Run inference on a few samples
    print("\nRunning inference on test data...")
    
    num_samples = 0
    with torch.no_grad():
        for batch_graphs in tqdm(test_loader):
            # Move data to device
            batch_graphs = [frame.to(device) for frame in batch_graphs]
            
            # Create agent mask
            agent_mask = create_agent_mask(batch_graphs, device)
            
            # Forward pass
            predictions = model(batch_graphs, agent_mask)
            
            # Print predictions shape
            print(f"\nSample predictions shape: {predictions.shape}")
            print(f"Value range: min={predictions.min().item():.4f}, max={predictions.max().item():.4f}")
            
            # Print a few sample predictions for the first agent in the first batch
            if predictions.shape[0] > 0 and predictions.shape[1] > 0:
                print("\nSample prediction for first agent:")
                first_agent_pred = predictions[0, 0]  # First batch, first agent
                for t in range(min(3, first_agent_pred.shape[0])):  # Show first 3 timesteps
                    print(f"  t+{t+1}: ({first_agent_pred[t, 0].item():.4f}, {first_agent_pred[t, 1].item():.4f})")
            
            num_samples += 1
            if num_samples >= 2:  # Only process a couple of batches for demonstration
                break
    
    print("\nModel loading and testing complete!")

if __name__ == "__main__":
    # Paths - adapt these to your saved model and data locations
    model_path = os.path.join('models', 'final_model.pt')
    data_path = 'final_surajpur_proper_reduced_2000.csv'
    
    # Load and test the model
    load_and_test_model(model_path, data_path) 