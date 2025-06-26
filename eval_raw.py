import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your model and data loader
from data_loader import RoundaboutTrajectoryDataLoader
from TCN_layer import TCN
from GAT_layer import GAT
from TRNS_layer import TransformerEncoder

def load_model(model_path, args):
    """Load the trained model"""
    # Define model architecture based on your existing architecture
    if args.model_type == 'tcn':
        model = TCN(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout
        )
    elif args.model_type == 'gat':
        model = GAT(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
    elif args.model_type == 'transformer':
        model = TransformerEncoder(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
        
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    return model

def calculate_metrics_raw(predictions, ground_truth, agent_mask=None):
    """
    Calculate ADE/FDE directly in the raw coordinate space without any normalization.
    
    Args:
        predictions: Model predictions [batch_size, num_agents, pred_len, 2]
        ground_truth: Ground truth future trajectories [batch_size, num_agents, pred_len, 2]
        agent_mask: Boolean mask for valid agents [batch_size, num_agents]
        
    Returns:
        Dictionary of metrics
    """
    # First, verify we have matching shapes
    if predictions.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, ground truth {ground_truth.shape}")
    
    # Print coordinate range information
    pred_min_x = predictions[..., 0].min().item()
    pred_max_x = predictions[..., 0].max().item()
    pred_min_y = predictions[..., 1].min().item()
    pred_max_y = predictions[..., 1].max().item()
    gt_min_x = ground_truth[..., 0].min().item()
    gt_max_x = ground_truth[..., 0].max().item()
    gt_min_y = ground_truth[..., 1].min().item()
    gt_max_y = ground_truth[..., 1].max().item()
    
    print(f"\nCoordinate ranges (raw):")
    print(f"Predictions X range: [{pred_min_x:.4f}, {pred_max_x:.4f}], span: {pred_max_x - pred_min_x:.4f}")
    print(f"Predictions Y range: [{pred_min_y:.4f}, {pred_max_y:.4f}], span: {pred_max_y - pred_min_y:.4f}")
    print(f"Ground Truth X range: [{gt_min_x:.4f}, {gt_max_x:.4f}], span: {gt_max_x - gt_min_x:.4f}")
    print(f"Ground Truth Y range: [{gt_min_y:.4f}, {gt_max_y:.4f}], span: {gt_max_y - gt_min_y:.4f}")
    
    # Get dimensions
    batch_size, num_agents, pred_len, _ = predictions.shape
    device = predictions.device
    
    # If no agent mask was provided, create one
    if agent_mask is None:
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool, device=device)
    
    # Initialize arrays for per-agent metrics
    all_ades = []
    all_fdes = []
    
    # Process each scene (batch) independently
    for b in range(batch_size):
        # Get valid agents for this scene
        scene_mask = agent_mask[b]
        valid_agents = torch.where(scene_mask)[0]
        
        # Skip if no valid agents
        if len(valid_agents) == 0:
            continue
            
        # Extract predictions and ground truth for valid agents only
        scene_pred = predictions[b, scene_mask]
        scene_gt = ground_truth[b, scene_mask]
        
        # Calculate Euclidean distance error at each timestep
        # Shape: [num_valid_agents, pred_len]
        error = torch.sqrt(((scene_pred - scene_gt) ** 2).sum(dim=-1) + 1e-10)
        
        # Final Displacement Error (FDE) - error at final predicted position
        scene_fde = error[:, -1]  # [num_valid_agents]
        
        # Average Displacement Error (ADE) - mean error across all timesteps
        scene_ade = error.mean(dim=1)  # [num_valid_agents]
        
        # Append to lists
        all_ades.append(scene_ade)
        all_fdes.append(scene_fde)
    
    # Concatenate metrics from all scenes
    if all_ades and all_fdes:
        all_ades = torch.cat(all_ades)
        all_fdes = torch.cat(all_fdes)
        
        # Calculate final metrics
        mean_ade = all_ades.mean().item()
        mean_fde = all_fdes.mean().item()
        
        # Debug: Print range of errors
        print(f"\nError statistics (raw):")
        print(f"ADE range: [{all_ades.min().item():.4f}, {all_ades.max().item():.4f}], mean: {mean_ade:.4f}")
        print(f"FDE range: [{all_fdes.min().item():.4f}, {all_fdes.max().item():.4f}], mean: {mean_fde:.4f}")
    else:
        # Handle edge case of no valid agents
        mean_ade = float('nan')
        mean_fde = float('nan')
        print("No valid agents found for metric calculation!")
    
    return {
        'ADE': mean_ade,
        'FDE': mean_fde,
    }

def visualize_trajectories(predictions, ground_truth, agent_idx=0, batch_idx=0):
    """
    Visualize the trajectories of a specific agent in a specific batch
    """
    plt.figure(figsize=(10, 8))
    
    # Get prediction and ground truth for specific agent
    pred_traj = predictions[batch_idx, agent_idx].detach().cpu().numpy()
    gt_traj = ground_truth[batch_idx, agent_idx].detach().cpu().numpy()
    
    # Plot trajectories
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', linewidth=2, marker='o', label='Prediction')
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', linewidth=2, marker='x', label='Ground Truth')
    
    # Add arrows to show direction
    plt.arrow(pred_traj[-2, 0], pred_traj[-2, 1], 
              pred_traj[-1, 0] - pred_traj[-2, 0], pred_traj[-1, 1] - pred_traj[-2, 1],
              head_width=0.05, head_length=0.1, fc='b', ec='b')
    plt.arrow(gt_traj[-2, 0], gt_traj[-2, 1], 
              gt_traj[-1, 0] - gt_traj[-2, 0], gt_traj[-1, 1] - gt_traj[-2, 1],
              head_width=0.05, head_length=0.1, fc='g', ec='g')
    
    plt.title(f"Trajectory Comparison (Agent {agent_idx}, Batch {batch_idx})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis('equal')  # Equal aspect ratio
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"trajectory_agent{agent_idx}_batch{batch_idx}.png", dpi=300)
    plt.close()

def compare_with_normalized():
    """
    Run evaluation with both normalized and raw coordinates
    """
    # Set up arguments (modify based on your trained model parameters)
    class Args:
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_type = 'transformer'  # Change to match your model: 'tcn', 'gat', 'transformer'
            self.input_dim = 5  # Adjust based on your actual input dimension
            self.hidden_dim = 64
            self.output_dim = 2  # x, y coordinates
            self.num_layers = 4
            self.num_heads = 8
            self.kernel_size = 3
            self.dropout = 0.1
            self.batch_size = 32
    
    args = Args()
    
    # Paths
    model_path = 'models/best_model.pth'  # Adjust to your model path
    data_path = 'final_surajpur_proper_reduced_2000.csv'  # Your data file
    
    # 1. Evaluation with normalization (your original setup)
    print("==========================================")
    print("EVALUATION WITH NORMALIZATION")
    print("==========================================")
    
    # Load dataset with normalization
    test_dataset_norm = RoundaboutTrajectoryDataLoader(
        csv_path=data_path,
        obs_len=10,
        pred_len=10,
        dist_threshold=10.0,
        standardize_xy=True  # With normalization
    )
    
    test_loader_norm = test_dataset_norm.get_loader(
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Load model
    model = load_model(model_path, args)
    
    # Run evaluation with normalization
    normalized_metrics = evaluate(model, test_loader_norm, args)
    
    # 2. Evaluation without normalization
    print("\n==========================================")
    print("EVALUATION WITHOUT NORMALIZATION")
    print("==========================================")
    
    # Load dataset without normalization
    test_dataset_raw = RoundaboutTrajectoryDataLoader(
        csv_path=data_path,
        obs_len=10,
        pred_len=10,
        dist_threshold=10.0,
        standardize_xy=False  # Without normalization
    )
    
    test_loader_raw = test_dataset_raw.get_loader(
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Run evaluation without normalization
    raw_metrics = evaluate(model, test_loader_raw, args)
    
    # Print comparison
    print("\n==========================================")
    print("METRICS COMPARISON")
    print("==========================================")
    print(f"Normalized: ADE = {normalized_metrics['ADE']:.4f}, FDE = {normalized_metrics['FDE']:.4f}")
    print(f"Raw:        ADE = {raw_metrics['ADE']:.4f}, FDE = {raw_metrics['FDE']:.4f}")
    
    return normalized_metrics, raw_metrics

def evaluate(model, test_loader, args):
    """
    Evaluate model on test data
    """
    device = args.device
    model.eval()
    
    total_ade = 0
    total_fde = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Process batch and generate predictions
            batch_data = [data.to(device) for data in batch_data]
            
            # Extract features (adjust based on your actual model input requirements)
            batch_size = len(batch_data[0].batch)
            obs_len = len(batch_data)
            node_features = []
            edge_indices = []
            type_ids = []
            
            for frame_data in batch_data:
                node_features.append(frame_data.x)
                edge_indices.append(frame_data.edge_index)
                type_ids.append(frame_data.type_ids)
            
            # Forward pass
            predictions = model(node_features, edge_indices, type_ids)
            
            # Reshape predictions to [batch_size, num_agents, pred_len, 2]
            # This part depends on your model's output format
            # The following is a placeholder, modify according to your model
            predictions = predictions.reshape(batch_size, -1, 10, 2)  # Assuming pred_len=10
            
            # Create ground truth tensor
            # This part depends on how your data is structured
            # The following is a placeholder, modify according to your specific dataset
            # For example, you might need to extract future positions from your dataset
            
            # Example placeholder for ground truth:
            ground_truth = torch.zeros_like(predictions)
            # You need to populate ground_truth with actual ground truth data
            # from your dataset
            
            # Calculate metrics
            metrics = calculate_metrics_raw(predictions, ground_truth)
            
            total_ade += metrics['ADE']
            total_fde += metrics['FDE']
            batch_count += 1
            
            # Visualize some trajectories
            if batch_idx == 0:
                for agent_idx in range(min(3, predictions.shape[1])):  # Visualize up to 3 agents
                    visualize_trajectories(predictions, ground_truth, agent_idx, 0)
    
    # Calculate average metrics
    avg_metrics = {
        'ADE': total_ade / batch_count,
        'FDE': total_fde / batch_count
    }
    
    return avg_metrics

if __name__ == "__main__":
    normalized_metrics, raw_metrics = compare_with_normalized() 