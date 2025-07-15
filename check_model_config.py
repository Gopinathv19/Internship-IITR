import os
import torch
import numpy as np

def check_model_config(model_path):
    """
    Check the configuration of a saved model without recreating it
    
    Args:
        model_path: Path to the saved model file
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the saved dictionary
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # Explicitly allow all objects for your own models
    
    # Extract configuration
    model_config = checkpoint.get('model_config', {})
    
    # Print model information
    print("\n" + "="*50)
    print("MODEL CONFIGURATION")
    print("="*50)
    
    if model_config:
        print("Model Hyperparameters:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
    else:
        print("No model configuration found in checkpoint")
    
    # Print training information
    print("\n" + "="*50)
    print("TRAINING INFORMATION")
    print("="*50)
    
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print("\nTraining Metrics:")
        if isinstance(metrics, dict):
            if 'train' in metrics:
                print("  Training metrics:")
                for key, value in metrics['train'].items():
                    print(f"    {key}: {value}")
            
            if 'val' in metrics:
                print("  Validation metrics:")
                for key, value in metrics['val'].items():
                    print(f"    {key}: {value}")
            
            if 'epoch' in metrics:
                print(f"  Epoch: {metrics['epoch']}")
            
            if 'time' in metrics:
                print(f"  Training time: {metrics['time']:.2f} seconds")
    
    # Print model state information
    print("\n" + "="*50)
    print("MODEL STATE INFORMATION")
    print("="*50)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Model state dict keys: {len(state_dict)}")
        print("Key examples:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {key}: {state_dict[key].shape}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more keys")
    
    if 'optimizer_state_dict' in checkpoint:
        print(f"Optimizer state dict present: Yes")
    
    if 'scheduler_state_dict' in checkpoint:
        print(f"Scheduler state dict present: Yes")
    
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    
    # Create a summary of the configuration
    if model_config:
        print("Model Architecture:")
        print(f"  GAT: hidden_dim={model_config.get('gat_hidden_dim', 'N/A')}, "
              f"output_dim={model_config.get('gat_output_dim', 'N/A')}, "
              f"heads={model_config.get('gat_heads', 'N/A')}")
        print(f"  TCN: hidden_dim={model_config.get('tcn_hidden_dim', 'N/A')}, "
              f"output_dim={model_config.get('tcn_output_dim', 'N/A')}, "
              f"layers={model_config.get('tcn_layers', 'N/A')}")
        print(f"  Transformer: dim={model_config.get('transformer_dim', 'N/A')}, "
              f"heads={model_config.get('transformer_heads', 'N/A')}, "
              f"encoder_layers={model_config.get('transformer_encoder_layers', 'N/A')}, "
              f"decoder_layers={model_config.get('transformer_decoder_layers', 'N/A')}")
        print(f"  Sequence: obs_len={model_config.get('obs_len', 'N/A')}, "
              f"pred_len={model_config.get('pred_len', 'N/A')}")
        print(f"  Dropout: {model_config.get('dropout', 'N/A')}")

if __name__ == "__main__":
    # Check the best model configuration
    model_path = os.path.join('models', 'best_model.pt')
    
    if os.path.exists(model_path):
        check_model_config(model_path)
    else:
        print(f"Model file not found: {model_path}")
        print("Available model files:")
        models_dir = 'models'
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pt') or file.endswith('.pth'):
                    print(f"  {file}") 