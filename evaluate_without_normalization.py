import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import time

# Import your model and data loader
from data_loader import RoundaboutTrajectoryDataLoader
from TCN_layer import TCN
from GAT_layer import SpatialGAT as GAT
from TRNS_layer import TrajectoryTransformer
from train import TrajectoryPredictionModel, calculate_metrics

def create_new_model(args):
    """Create a new model instance instead of loading"""
    model = TrajectoryPredictionModel(args)
    model.to(args.device)
    model.eval()
    print("Created new model instance for evaluation")
    return model

def calculate_raw_metrics(predictions, ground_truth, agent_mask=None):
    """
    Calculate ADE/FDE directly in the original coordinate space without any normalization.
    
    Args:
        predictions: Model predictions [batch_size, num_agents, pred_len, 2]
        ground_truth: Ground truth future trajectories [batch_size, num_agents, pred_len, 2]
        agent_mask: Boolean mask for valid agents [batch_size, num_agents]
    """
    # First, verify matching shapes
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
    
    # Calculate metrics for each scene
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
        
        # Print range of errors
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

def visualize_trajectories(predictions, ground_truth, agent_idx=0, batch_idx=0, save_path='trajectory_comparison.png'):
    """
    Visualize the predicted vs ground truth trajectories
    """
    plt.figure(figsize=(10, 8))
    
    # Get prediction and ground truth for specific agent
    pred_traj = predictions[batch_idx, agent_idx].detach().cpu().numpy()
    gt_traj = ground_truth[batch_idx, agent_idx].detach().cpu().numpy()
    
    # Plot trajectories
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', linewidth=2, marker='o', label='Prediction')
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', linewidth=2, marker='x', label='Ground Truth')
    
    # Add arrows to show direction
    if pred_traj.shape[0] >= 2:
        plt.arrow(pred_traj[-2, 0], pred_traj[-2, 1], 
                pred_traj[-1, 0] - pred_traj[-2, 0], pred_traj[-1, 1] - pred_traj[-2, 1],
                head_width=0.5, head_length=1.0, fc='b', ec='b')
    
    if gt_traj.shape[0] >= 2:
        plt.arrow(gt_traj[-2, 0], gt_traj[-2, 1], 
                gt_traj[-1, 0] - gt_traj[-2, 0], gt_traj[-1, 1] - gt_traj[-2, 1],
                head_width=0.5, head_length=1.0, fc='g', ec='g')
    
    plt.title(f"Trajectory Comparison (Agent {agent_idx}, Batch {batch_idx})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis('equal')  # Equal aspect ratio
    plt.legend()
    plt.grid(True)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_model(args):
    """
    Run evaluation with both normalized and raw coordinates
    """
    device = args.device
    
    # Make sure models directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Evaluation with normalization (original)
    print("="*50)
    print("EVALUATION WITH NORMALIZATION")
    print("="*50)
    
    # Load dataset with normalization
    normal_dataset = RoundaboutTrajectoryDataLoader(
        csv_path=args.data_path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        dist_threshold=args.dist_threshold,
        standardize_xy=True  # With normalization
    )
    
    normal_loader = normal_dataset.get_loader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create a new model (no loading)
    model = create_new_model(args)
    
    # Run evaluation with normalization
    normalized_metrics = run_evaluation(
        model, 
        normal_loader, 
        device, 
        args, 
        convert_coordinates=normal_dataset.inverse_transform_coordinates,
        output_prefix="norm"
    )
    
    # 2. Evaluation without normalization
    print("\n" + "="*50)
    print("EVALUATION WITHOUT NORMALIZATION")
    print("="*50)
    
    # Load dataset without normalization
    raw_dataset = RoundaboutTrajectoryDataLoader(
        csv_path=args.data_path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        dist_threshold=args.dist_threshold,
        standardize_xy=False  # Without normalization
    )
    
    raw_loader = raw_dataset.get_loader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Run evaluation without normalization
    raw_metrics = run_evaluation(
        model, 
        raw_loader, 
        device, 
        args,
        convert_coordinates=None,  # No coordinate conversion needed
        output_prefix="raw"
    )
    
    # Print comparison
    print("\n" + "="*50)
    print("METRICS COMPARISON")
    print("="*50)
    print(f"Normalized: ADE = {normalized_metrics['ADE']:.4f}, FDE = {normalized_metrics['FDE']:.4f}")
    print(f"Raw:        ADE = {raw_metrics['ADE']:.4f}, FDE = {raw_metrics['FDE']:.4f}")
    
    # Calculate ratio if we have valid metrics
    if normalized_metrics['ADE'] != 0 and normalized_metrics['FDE'] != 0:
        print(f"Ratio:      ADE = {raw_metrics['ADE']/normalized_metrics['ADE']:.2f}x, FDE = {raw_metrics['FDE']/normalized_metrics['FDE']:.2f}x")
    
    # Save the model for future use
    model_path = os.path.join(args.save_dir, 'new_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    return normalized_metrics, raw_metrics

def run_evaluation(model, data_loader, device, args, convert_coordinates=None, output_prefix=""):
    """
    Run evaluation on the provided data loader
    """
    model.eval()
    
    with torch.no_grad():
        # Process a single batch for demonstration
        for batch_idx, batch_graphs in enumerate(data_loader):
            # Move batch to device
            batch_graphs = [data.to(device) for data in batch_graphs]
            
            # Create agent mask for tracking valid agents
            from train import create_agent_mask
            agent_mask = create_agent_mask(batch_graphs, device)
            
            # Forward pass through the model
            predictions = model(batch_graphs, agent_mask)
            
            # Extract ground truth future trajectories
            from train import extract_future_data
            gt_trajectories = extract_future_data(batch_graphs, args.pred_len, data_loader)
            
            # Calculate metrics using both methods
            print("\nUsing standard metrics function:")
            metrics1 = calculate_metrics(
                predictions, 
                gt_trajectories, 
                agent_mask=agent_mask,
                convert_coordinates=convert_coordinates
            )
            
            print("\nUsing raw metrics function (direct calculation):")
            # If we have coordinate conversion, apply it first
            if convert_coordinates is not None:
                pred_orig = convert_coordinates(predictions)
                gt_orig = convert_coordinates(gt_trajectories)
                metrics2 = calculate_raw_metrics(
                    pred_orig, 
                    gt_orig,
                    agent_mask=agent_mask
                )
            else:
                metrics2 = calculate_raw_metrics(
                    predictions,
                    gt_trajectories,
                    agent_mask=agent_mask
                )
            
            # Visualize some trajectories
            if batch_idx == 0:
                for agent_idx in range(min(3, predictions.shape[1])):  # Visualize up to 3 agents
                    if convert_coordinates is not None:
                        # Visualize in original coordinate space if conversion is available
                        vis_pred = convert_coordinates(predictions)
                        vis_gt = convert_coordinates(gt_trajectories)
                    else:
                        vis_pred = predictions
                        vis_gt = gt_trajectories
                        
                    save_path = f"{args.save_dir}/{output_prefix}_trajectory_agent{agent_idx}_batch{batch_idx}.png"
                    visualize_trajectories(vis_pred, vis_gt, agent_idx, 0, save_path)
            
            # We only process one batch for demonstration
            return metrics2
            
    # If we didn't return from the loop (no batches), return dummy metrics
    return {'ADE': float('nan'), 'FDE': float('nan')}

if __name__ == "__main__":
    # Define arguments similar to the model training configuration
    class Args:
        def __init__(self):
            # Data parameters
            self.data_path = 'final_surajpur_proper_reduced_2000.csv'
            self.obs_len = 10
            self.pred_len = 10
            self.dist_threshold = 10.0
            self.batch_size = 8
            
            # Model parameters (must match your trained model)
            self.num_types = 9
            self.type_embed_dim = 16
            
            self.gat_hidden_dim = 64
            self.gat_output_dim = 64
            self.gat_heads = 2
            
            self.tcn_hidden_dim = 128
            self.tcn_output_dim = 128
            self.tcn_layers = 3
            self.tcn_kernel_size = 3
            
            self.transformer_dim = 256
            self.transformer_heads = 8
            self.transformer_encoder_layers = 4
            self.transformer_decoder_layers = 4
            self.transformer_ffn_dim = 1024
            
            self.dropout = 0.1
            
            # Other parameters
            self.save_dir = 'models'
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args = Args()
    print(f"Running evaluation on device: {args.device}")
    start_time = time.time()
    normalized_metrics, raw_metrics = evaluate_model(args)
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds") 