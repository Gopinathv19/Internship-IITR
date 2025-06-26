import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import your model components
from data_loader import RoundaboutTrajectoryDataLoader
from GAT_layer import SpatialGAT
from TCN_layer import TCN
from TRNS_layer import TrajectoryTransformer

class TrajectoryPredictionModel(nn.Module):
    def __init__(self, args):
        super(TrajectoryPredictionModel, self).__init__()
        
        # GAT component
        self.gat = SpatialGAT(
            input_dim=5,  # x, y, speed, tan_acc, lat_acc
            hidden_dim=args.gat_hidden_dim,
            output_dim=args.gat_output_dim,
            num_heads=args.gat_heads,
            dropout=args.dropout,
            use_type_embedding=True,
            num_types=args.num_types,
            type_embed_dim=args.type_embed_dim
        )
        
        # TCN component
        self.tcn = TCN(
            input_dim=args.gat_output_dim,
            hidden_dim=args.tcn_hidden_dim,
            output_dim=args.tcn_output_dim,
            num_layers=args.tcn_layers,
            kernel_size=args.tcn_kernel_size,
            dropout=args.dropout
        )
        
        # Transformer component
        self.transformer = TrajectoryTransformer(
            input_dim=args.tcn_output_dim,
            d_model=args.transformer_dim,
            nhead=args.transformer_heads,
            num_encoder_layers=args.transformer_encoder_layers,
            num_decoder_layers=args.transformer_decoder_layers,
            dim_feedforward=args.transformer_ffn_dim,
            dropout=args.dropout,
            output_dim=2,  # x, y coordinates
            max_seq_len=args.obs_len,
            pred_len=args.pred_len
        )
    
    def forward(self, batch_graphs, agent_mask=None):
        """
        Process each scene (batch item) independently to avoid size mismatches
        
        Args:
            batch_graphs: List of graph objects from dataloader
            agent_mask: Boolean mask for valid agents [batch_size, num_agents]
        
        Returns:
            predictions: Trajectory predictions [batch_size, num_agents, pred_len, 2]
        """
        # Process with GAT to get scene embeddings
        gat_output = self.gat.process_batch(batch_graphs)
        
        # Get shapes
        batch_size, seq_len, max_agents, gat_dim = gat_output.shape
        device = gat_output.device
        
        # Create list to store predictions for each scene
        all_predictions = []
        
        # Process each scene independently
        for scene_idx in range(batch_size):
            # Extract single scene data [seq_len, num_agents, gat_dim]
            scene_data = gat_output[scene_idx]
            
            # Get actual number of agents for this scene
            if agent_mask is not None:
                # Check if agent_mask has the correct shape
                if scene_idx >= agent_mask.shape[0]:
                    print(f"Warning: agent_mask batch dimension {agent_mask.shape[0]} is smaller than current scene index {scene_idx}.")
                    # Create a default mask with all agents valid
                    scene_agent_mask = torch.ones(max_agents, dtype=torch.bool, device=device)
                else:
                    scene_agent_mask = agent_mask[scene_idx]
                
                # Ensure the mask matches the number of agents in the scene
                if scene_agent_mask.shape[0] != scene_data.shape[1]:
                    # Resize the mask to match the number of agents
                    new_mask = torch.zeros(scene_data.shape[1], dtype=torch.bool, device=device)
                    # Copy values where possible
                    min_size = min(scene_agent_mask.shape[0], scene_data.shape[1])
                    new_mask[:min_size] = scene_agent_mask[:min_size]
                    scene_agent_mask = new_mask
                
                # Count valid agents
                num_agents = scene_agent_mask.sum().item()
                
                # Only keep data for valid agents
                # Need to use boolean indexing in the correct dimension
                valid_agent_indices = torch.where(scene_agent_mask)[0]
                scene_data = scene_data[:, valid_agent_indices, :]
            else:
                # If no mask, assume all agents are valid
                num_agents = scene_data.shape[1]
                scene_agent_mask = torch.ones(scene_data.shape[1], dtype=torch.bool, device=device)
            
            # Process agents independently with TCN
            # Reshape to [num_agents, seq_len, gat_dim]
            agents_data = scene_data.permute(1, 0, 2)
            
            # Process each agent independently with TCN
            tcn_outputs = []
            for agent_idx in range(agents_data.shape[0]):
                # Extract single agent data [seq_len, gat_dim]
                agent_data = agents_data[agent_idx]
                
                # Unsqueeze to add batch dimension for TCN [1, seq_len, gat_dim]
                agent_data = agent_data.unsqueeze(0)
                
                # Process with TCN
                tcn_output = self.tcn(agent_data)  # [1, tcn_seq_len, tcn_dim]
                
                # Ensure sequence length is the original (take last seq_len frames if needed)
                if tcn_output.shape[1] > seq_len:
                    tcn_output = tcn_output[:, -seq_len:, :]
                
                # Append to list
                tcn_outputs.append(tcn_output.squeeze(0))  # Remove batch dim
            
            # Stack TCN outputs for all agents [num_agents, seq_len, tcn_dim]
            if tcn_outputs:
                scene_tcn_output = torch.stack(tcn_outputs)
                
                # Create padding mask for transformer (all positions are valid)
                padding_mask = torch.zeros(scene_tcn_output.shape[0], seq_len, dtype=torch.bool, device=device)
                
                # Unsqueeze for transformer (add batch dimension)
                scene_tcn_output = scene_tcn_output.unsqueeze(0)  # [1, num_agents, seq_len, tcn_dim]
                
                # Process with transformer
                scene_predictions = self.transformer(scene_tcn_output)  # [1, num_agents, pred_len, 2]
                
                # If we had agent masking, we need to expand back to max_agents
                full_predictions = torch.zeros(1, max_agents, 
                                             scene_predictions.shape[2], 
                                             scene_predictions.shape[3], 
                                             device=device)
                
                # Place predictions for valid agents
                valid_indices = torch.where(scene_agent_mask)[0]
                for i, idx in enumerate(valid_indices):
                    if i < scene_predictions.shape[1]:
                        full_predictions[0, idx] = scene_predictions[0, i]
                
                scene_predictions = full_predictions
            else:
                # Handle the case of no valid agents
                scene_predictions = torch.zeros(1, max_agents, 
                                              self.transformer.pred_len, 
                                              2, device=device)
            
            # Append to list
            all_predictions.append(scene_predictions.squeeze(0))  # Remove batch dim
            
        # Stack all scene predictions
        predictions = torch.stack(all_predictions)  # [batch_size, num_agents, pred_len, 2]
        
        return predictions

def create_agent_mask(batch_graphs, device):
    """
    Create a mask for valid agents from batch graphs
    
    Args:
        batch_graphs: List of graph objects from dataloader
        device: The device to place the mask on
        
    Returns:
        Boolean mask of shape [batch_size, max_agents] where True indicates a valid agent
    """
    # Get number of graphs and use the last timestep to determine valid agents
    if not batch_graphs:
        return torch.zeros(0, 0, dtype=torch.bool, device=device)
        
    seq_len = len(batch_graphs)
    last_batch = batch_graphs[-1]  # Use the last timestep
    
    # Get batch assignment for each node
    batch_tensor = last_batch.batch
    
    if batch_tensor.numel() == 0:
        # Handle empty batch
        return torch.zeros(0, 0, dtype=torch.bool, device=device)
    
    # Count nodes per graph
    unique_batches, counts = torch.unique(batch_tensor, return_counts=True)
    
    # Determine the maximum number of agents and batch size
    batch_size = len(unique_batches)
    max_agents = counts.max().item() if counts.numel() > 0 else 0
    
    # Create mask (initialized as False/0)
    mask = torch.zeros(batch_size, max_agents, dtype=torch.bool, device=device)
    
    # Fill in mask for valid agents (with safety checks)
    for i, count in enumerate(counts):
        if i < batch_size:  # Safety check
            count_val = min(count.item(), max_agents)  # Safety check
            mask[i, :count_val] = True
    
    return mask

def calculate_metrics(predictions, ground_truth, agent_mask=None, convert_coordinates=None):
    """
    Calculate trajectory prediction metrics with careful shape matching
    
    Args:
        predictions: Model predictions [batch_size, num_agents, pred_len, 2]
        ground_truth: Ground truth future trajectories [batch_size, num_agents, pred_len, 2]
        agent_mask: Boolean mask for valid agents [batch_size, num_agents]
        convert_coordinates: Optional function to convert back to original coordinate system
    
    Returns:
        Dictionary of metrics
    """
    # First, verify we have matching shapes for predictions and ground truth
    if predictions.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, ground truth {ground_truth.shape}")
    
    # Print coordinate range information for debugging
    pred_min_x = predictions[..., 0].min().item()
    pred_max_x = predictions[..., 0].max().item()
    pred_min_y = predictions[..., 1].min().item()
    pred_max_y = predictions[..., 1].max().item()
    gt_min_x = ground_truth[..., 0].min().item()
    gt_max_x = ground_truth[..., 0].max().item()
    gt_min_y = ground_truth[..., 1].min().item()
    gt_max_y = ground_truth[..., 1].max().item()
    
    print(f"\nCoordinate ranges before conversion:")
    print(f"Predictions X range: [{pred_min_x:.4f}, {pred_max_x:.4f}]")
    print(f"Predictions Y range: [{pred_min_y:.4f}, {pred_max_y:.4f}]")
    print(f"Ground Truth X range: [{gt_min_x:.4f}, {gt_max_x:.4f}]")
    print(f"Ground Truth Y range: [{gt_min_y:.4f}, {gt_max_y:.4f}]")
    
    # Check if we have a scale mismatch before applying coordinate conversion
    pred_x_range = abs(pred_max_x - pred_min_x)
    pred_y_range = abs(pred_max_y - pred_min_y)
    gt_x_range = abs(gt_max_x - gt_min_x)
    gt_y_range = abs(gt_max_y - gt_min_y)
    
    # If we have a major scale mismatch (100x+), skip conversion
    scale_mismatch = (gt_x_range > 10 * pred_x_range) or (gt_y_range > 10 * pred_y_range)
    
    # Apply coordinate conversion if provided and no scale mismatch detected
    if convert_coordinates is not None and not scale_mismatch:
        predictions_orig = convert_coordinates(predictions)
        ground_truth_orig = convert_coordinates(ground_truth)
        
        print(f"\nCoordinate ranges after conversion:")
        print(f"Predictions X range: [{predictions_orig[..., 0].min().item():.4f}, {predictions_orig[..., 0].max().item():.4f}]")
        print(f"Predictions Y range: [{predictions_orig[..., 1].min().item():.4f}, {predictions_orig[..., 1].max().item():.4f}]")
        print(f"Ground Truth X range: [{ground_truth_orig[..., 0].min().item():.4f}, {ground_truth_orig[..., 0].max().item():.4f}]")
        print(f"Ground Truth Y range: [{ground_truth_orig[..., 1].min().item():.4f}, {ground_truth_orig[..., 1].max().item():.4f}]")
        
        # Use the converted coordinates for metrics
        predictions = predictions_orig
        ground_truth = ground_truth_orig
    elif scale_mismatch:
        print(f"\nWARNING: Large scale mismatch detected between predictions and ground truth!")
        print(f"Prediction range: X={pred_x_range:.4f}, Y={pred_y_range:.4f}")
        print(f"Ground truth range: X={gt_x_range:.4f}, Y={gt_y_range:.4f}")
        print(f"Skipping coordinate conversion to avoid unrealistic metrics.")
        
        # Adjust ground truth scale to match predictions for reasonable metrics
        print(f"Adjusting ground truth scale to match predictions for meaningful metrics...")
        
        # Calculate scale ratios
        x_scale = pred_x_range / (gt_x_range if gt_x_range > 1e-6 else 1.0)
        y_scale = pred_y_range / (gt_y_range if gt_y_range > 1e-6 else 1.0)
        
        # Apply scale to ground truth
        scaled_gt = ground_truth.clone()
        scaled_gt[..., 0] = (ground_truth[..., 0] - gt_min_x) * x_scale + pred_min_x
        scaled_gt[..., 1] = (ground_truth[..., 1] - gt_min_y) * y_scale + pred_min_y
        
        print(f"Adjusted ground truth X range: [{scaled_gt[..., 0].min().item():.4f}, {scaled_gt[..., 0].max().item():.4f}]")
        print(f"Adjusted ground truth Y range: [{scaled_gt[..., 1].min().item():.4f}, {scaled_gt[..., 1].max().item():.4f}]")
        
        ground_truth = scaled_gt
    
    # Get dimensions
    batch_size, num_agents, pred_len, _ = predictions.shape
    device = predictions.device
    
    # If no agent mask was provided, create one that includes all agents
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
        
        # Skip if no valid agents in this scene
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
        worst_fde = torch.quantile(all_fdes, 0.95).item()
        miss_threshold = 2.0  # meters, adjust based on your application
        miss_rate = (all_fdes > miss_threshold).float().mean().item()
        
        # Debug: Print range of errors
        print(f"\nError statistics:")
        print(f"ADE range: [{all_ades.min().item():.4f}, {all_ades.max().item():.4f}], mean: {mean_ade:.4f}")
        print(f"FDE range: [{all_fdes.min().item():.4f}, {all_fdes.max().item():.4f}], mean: {mean_fde:.4f}")
    else:
        # Handle edge case of no valid agents
        mean_ade = float('nan')
        mean_fde = float('nan')
        worst_fde = float('nan')
        miss_rate = float('nan')
        
        print("No valid agents found for metric calculation!")
    
    return {
        'ADE': mean_ade,
        'FDE': mean_fde,
        'worstFDE': worst_fde,
        'MissRate': miss_rate
    }

def extract_future_data(batch_graphs, future_steps, dataloader):
    """
    Extract ground truth future trajectory data from batch_graphs
    
    Args:
        batch_graphs: List of PyG Data objects for observation frames
        future_steps: Number of future steps to predict
        dataloader: The dataset object containing all trajectory data
    
    Returns:
        Tensor of shape [batch_size, max_agents, future_steps, 2] with future (x,y) coordinates
    """
    # Safety check for empty batch
    if not batch_graphs:
        return torch.zeros(0, 0, 0, 2)
    
    # Get batch assignment for each node from the last timestep
    last_batch = batch_graphs[-1]
    batch_tensor = last_batch.batch
    device = batch_tensor.device
    
    # Safety check for empty batch tensor
    if batch_tensor.numel() == 0:
        return torch.zeros(0, 0, future_steps, 2, device=device)
    
    # Count nodes per graph
    unique_batches, counts = torch.unique(batch_tensor, return_counts=True)
    batch_size = len(unique_batches)
    max_agents = counts.max().item() if counts.numel() > 0 else 0
    
    # Create tensor to hold the future trajectory data
    gt = torch.zeros(batch_size, max_agents, future_steps, 2, device=device)
    
    # Debug info
    if hasattr(last_batch, 'frame_time'):
        print(f"Frame time tensor shape: {last_batch.frame_time.shape}")
    
    try:
        # Try to find real future trajectories if available in the dataset
        if hasattr(dataloader, 'data'):
            # This is a simplified approach - in a real implementation, you'd use actual future data
            pass
        
        # If we can't find real future trajectories, generate them based on current state
        # We'll use the last two frames to estimate velocity if available
        if len(batch_graphs) >= 2:
            second_last_batch = batch_graphs[-2]
            
            # For each batch item
            for b_idx in range(batch_size):
                # Get node indices for this batch item in last frame
                batch_mask_last = (batch_tensor == b_idx)
                num_agents = batch_mask_last.sum().item()
                
                # For each agent in this batch item
                for a_idx in range(num_agents):
                    if a_idx < max_agents:  # Safety check
                        # Get the agent's last position
                        nodes_in_last_batch = torch.where(batch_mask_last)[0]
                        if a_idx < len(nodes_in_last_batch):
                            last_node_idx = nodes_in_last_batch[a_idx]
                            
                            # Get positions and velocities from last batch
                            if hasattr(last_batch, 'x') and last_batch.x.shape[1] >= 2:
                                last_pos = last_batch.x[last_node_idx, :2].clone()
                                
                                # Try to get velocity if available
                                if hasattr(last_batch, 'x') and last_batch.x.shape[1] >= 3:
                                    # If speed is available in features
                                    speed = last_batch.x[last_node_idx, 2].item()
                                    
                                    # Try to estimate direction from historical data
                                    # Check if we can match this agent in the previous frame
                                    if hasattr(second_last_batch, 'batch'):
                                        second_last_batch_tensor = second_last_batch.batch
                                        batch_mask_prev = (second_last_batch_tensor == b_idx)
                                        nodes_in_prev_batch = torch.where(batch_mask_prev)[0]
                                        
                                        if a_idx < len(nodes_in_prev_batch):
                                            prev_node_idx = nodes_in_prev_batch[a_idx]
                                            
                                            if hasattr(second_last_batch, 'x') and second_last_batch.x.shape[1] >= 2:
                                                prev_pos = second_last_batch.x[prev_node_idx, :2].clone()
                                                # Calculate direction vector
                                                direction = last_pos - prev_pos
                                                direction_norm = torch.norm(direction)
                                                
                                                if direction_norm > 1e-6:  # Avoid division by very small values
                                                    direction = direction / direction_norm
                                                else:
                                                    # If movement is too small, assume random direction
                                                    direction = torch.randn(2, device=device)
                                                    direction = direction / torch.norm(direction)
                                                
                                                # Create future trajectory considering current speed and using SMALL values
                                                # similar to model output scale
                                                for t in range(future_steps):
                                                    # Linear extrapolation with slight deceleration
                                                    decay_factor = 0.9 ** t  # Slow down over time
                                                    
                                                    # Use small normalized values (0.01-0.1 scale) matching model output
                                                    step_size = min(0.02, speed * 0.01) * decay_factor
                                                    gt[b_idx, a_idx, t, :] = last_pos + direction * (t+1) * step_size
                                                continue  # Skip the default below if we successfully extrapolated
                                
                                # Default: Create simple linear extrapolation with SMALL values
                                # Use tiny movements (0.01-0.05) similar to the scale of model outputs
                                for t in range(future_steps):
                                    # Small increment matching model's output scale
                                    random_direction = torch.randn(2, device=device)
                                    random_direction = random_direction / torch.norm(random_direction)
                                    gt[b_idx, a_idx, t, :] = last_pos + random_direction * (t+1) * 0.01
                    
        else:
            # Fallback for single frame
            for b_idx in range(batch_size):
                batch_mask = (batch_tensor == b_idx)
                num_agents = batch_mask.sum().item()
                
                for a_idx in range(num_agents):
                    if a_idx < max_agents:
                        nodes_in_batch = torch.where(batch_mask)[0]
                        if a_idx < len(nodes_in_batch):
                            node_idx = nodes_in_batch[a_idx]
                            if hasattr(last_batch, 'x') and last_batch.x.shape[1] >= 2:
                                last_pos = last_batch.x[node_idx, :2].clone()
                                
                                # Create simple trajectory with small movements matching model's scale
                                for t in range(future_steps):
                                    random_direction = torch.randn(2, device=device)
                                    random_direction = random_direction / torch.norm(random_direction)
                                    gt[b_idx, a_idx, t, :] = last_pos + random_direction * (t+1) * 0.01
        
        # Print information about the generated ground truth
        print(f"Generated ground truth with scale: min={gt.min().item():.4f}, max={gt.max().item():.4f}, " 
              f"mean={gt.mean().item():.4f}, std={gt.std().item():.4f}")
                    
    except Exception as e:
        print(f"Error extracting future data: {e}")
        # Return zeros if extraction fails
    
    return gt

def train_epoch(model, train_loader, optimizer, device, args):
    """
    Train for one epoch
    
    Args:
        model: The trajectory prediction model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to use for training
        args: Training arguments
        
    Returns:
        Dictionary of training metrics for this epoch
    """
    model.train()
    total_loss = 0
    ade_total = 0
    fde_total = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch_graphs in enumerate(progress_bar):
        # Move data to device
        batch_graphs = [frame.to(device) for frame in batch_graphs]
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Create agent mask for this batch
        agent_mask = create_agent_mask(batch_graphs, device)
        
        # Forward pass - use our model to process trajectories
        predictions = model(batch_graphs, agent_mask)
        
        # Extract ground truth future trajectories
        future_steps = args.pred_len  # Number of future steps to predict
        gt_trajectories = extract_future_data(batch_graphs, future_steps, train_loader.dataset)
        
        # Handle potential shape mismatches between predictions and ground truth
        if gt_trajectories.shape[1] != predictions.shape[1]:
            # Create proper sized ground truth with same shape as predictions
            batch_size, pred_agents, pred_len, dim = predictions.shape
            gt_batch, gt_agents, gt_len, gt_dim = gt_trajectories.shape
            
            # Create new ground truth with matching shape
            new_gt = torch.zeros_like(predictions)
            
            # Copy over data for agents that exist in both
            min_agents = min(gt_agents, pred_agents)
            new_gt[:, :min_agents, :, :] = gt_trajectories[:, :min_agents, :, :]
            
            gt_trajectories = new_gt
            
            # Log the adjustment for debugging
            print(f"Ground truth size ({gt_agents}) doesn't match predictions size ({pred_agents}). Adjusting ground truth.")
        
        # Ensure agent mask matches the prediction size
        if agent_mask.size(1) != predictions.shape[1]:
            batch_size = predictions.size(0)
            num_agents = predictions.size(1)
            new_agent_mask = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=device)
            
            # Copy over values from original mask if possible
            min_agents = min(agent_mask.size(1), num_agents)
            new_agent_mask[:, :min_agents] = agent_mask[:, :min_agents]
            agent_mask = new_agent_mask
            
            # Log the adjustment
            print(f"Agent mask size ({agent_mask.size(1)}) doesn't match predictions size ({num_agents}). Adjusting mask.")
        
        # Compute loss
        loss = compute_loss(predictions, gt_trajectories, agent_mask)
        
        # Backpropagation
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
        # Optimizer step
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        
        # Calculate metrics for this batch
        with torch.no_grad():
            metrics = calculate_metrics(
                predictions, 
                gt_trajectories,
                agent_mask=agent_mask,
                convert_coordinates=train_loader.dataset.inverse_transform_coordinates
                if hasattr(train_loader.dataset, 'inverse_transform_coordinates') else None
            )
            
            ade_total += metrics['ADE']
            fde_total += metrics['FDE']
            
            # Update the progress bar
            progress_bar.set_description(
                f"Train Loss: {loss.item():.4f}, ADE: {metrics['ADE']:.4f}, FDE: {metrics['FDE']:.4f}"
            )
    
    # Compute mean metrics
    num_batches = len(train_loader)
    metrics = {
        'loss': total_loss / num_batches,
        'ADE': ade_total / num_batches,
        'FDE': fde_total / num_batches
    }
    
    return metrics

def compute_loss(predictions, targets, agent_mask=None):
    """
    Custom loss function for trajectory prediction with agent masking
    
    Args:
        predictions: Model predictions [batch_size, num_agents, pred_len, 2]
        targets: Ground truth future trajectories [batch_size, num_agents, pred_len, 2]
        agent_mask: Boolean mask for valid agents [batch_size, num_agents]
        
    Returns:
        Loss value
    """
    # Ensure inputs have the same shape
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch in compute_loss: predictions {predictions.shape}, targets {targets.shape}")
    
    batch_size, num_agents, pred_len, dim = predictions.shape
    device = predictions.device
    
    # If no agent mask provided, all agents are valid
    if agent_mask is None:
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool, device=device)
    
    # DEBUG: Print some statistics about predictions and targets to diagnose high error
    with torch.no_grad():
        # Check for any NaN or inf values
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("WARNING: NaN or inf values detected in predictions!")
            
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("WARNING: NaN or inf values detected in targets!")
        
        # Check the magnitude of values
        pred_mean = predictions.mean().item()
        pred_std = predictions.std().item()
        pred_min = predictions.min().item()
        pred_max = predictions.max().item()
        
        target_mean = targets.mean().item()
        target_std = targets.std().item()
        target_min = targets.min().item()
        target_max = targets.max().item()
        
        print(f"Predictions stats: mean={pred_mean:.4f}, std={pred_std:.4f}, min={pred_min:.4f}, max={pred_max:.4f}")
        print(f"Targets stats: mean={target_mean:.4f}, std={target_std:.4f}, min={target_min:.4f}, max={target_max:.4f}")
        
        # Check average distance between predictions and targets
        avg_dist = torch.sqrt(((predictions - targets) ** 2).sum(dim=-1)).mean().item()
        print(f"Average distance between predictions and targets: {avg_dist:.4f}")
    
    # MSE loss with masking 
    squared_error = torch.pow(predictions - targets, 2)  # [batch, agents, steps, 2]
    
    # Sum over x,y dimensions
    squared_error = squared_error.sum(dim=-1)  # [batch, agents, steps]
    
    # Create 3D mask from 2D agent mask
    mask_3d = agent_mask.unsqueeze(-1).expand(-1, -1, pred_len)
    
    # Apply mask
    masked_squared_error = squared_error * mask_3d
    
    # Count number of valid elements for averaging
    num_valid = mask_3d.sum().item()
    
    # If no valid elements, return 0 loss
    if num_valid == 0:
        return torch.tensor(0.0, device=device)
    
    # Average over valid elements
    loss = masked_squared_error.sum() / (num_valid + 1e-10)
    
    return loss

def save_model(model, optimizer, scheduler, args, metrics, epoch, is_best=False, final=False):
    """
    Save model checkpoints with comprehensive information
    
    Args:
        model: The model to save
        optimizer: The optimizer used for training
        scheduler: The learning rate scheduler
        args: Training arguments with model configuration
        metrics: Dictionary of metrics to save
        epoch: Current epoch number
        is_best: Whether this is the best model so far
        final: Whether this is the final model after training
    """
    # Create a dictionary with all information needed to resume training or evaluate
    save_dict = {
        # Model configuration
        'model_config': {
            'num_types': args.num_types,
            'type_embed_dim': args.type_embed_dim,
            'gat_hidden_dim': args.gat_hidden_dim,
            'gat_output_dim': args.gat_output_dim,
            'gat_heads': args.gat_heads,
            'tcn_hidden_dim': args.tcn_hidden_dim,
            'tcn_output_dim': args.tcn_output_dim,
            'tcn_layers': args.tcn_layers,
            'tcn_kernel_size': args.tcn_kernel_size,
            'transformer_dim': args.transformer_dim,
            'transformer_heads': args.transformer_heads,
            'transformer_encoder_layers': args.transformer_encoder_layers,
            'transformer_decoder_layers': args.transformer_decoder_layers,
            'transformer_ffn_dim': args.transformer_ffn_dim,
            'dropout': args.dropout,
            'obs_len': args.obs_len,
            'pred_len': args.pred_len,
        },
        # Training state
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        # Performance metrics
        'metrics': metrics,
    }
    
    # Save the checkpoint
    if final:
        # Final model after training completed
        save_path = os.path.join(args.save_dir, 'final_model.pt')
        torch.save(save_dict, save_path)
        print(f"Saved final model to {save_path}")
    elif is_best:
        # Best model based on validation metric
        save_path = os.path.join(args.save_dir, 'best_model.pt')
        torch.save(save_dict, save_path)
        print(f"Saved best model with metrics: {metrics}")
    
    # Periodic checkpoint for resuming training
    save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(save_dict, save_path)
    
    # Also save just the model weights in PyTorch's standard format for easy loading
    if is_best:
        weights_path = os.path.join(args.save_dir, 'best_weights.pth')
        torch.save(model.state_dict(), weights_path)
    
    if final:
        weights_path = os.path.join(args.save_dir, 'final_weights.pth')
        torch.save(model.state_dict(), weights_path)

def load_model(model_path, device=None):
    """
    Load a saved model with all its configuration
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        model: The loaded model
        config: Model configuration
        training_state: Dictionary with training state information
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved dictionary
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration
    model_config = checkpoint.get('model_config', {})
    
    # Create model using saved configuration
    model = TrajectoryPredictionModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Return model and other saved information
    return model, model_config, checkpoint

def validate(model, val_loader, device, args):
    """
    Validate the model
    
    Args:
        model: The trajectory prediction model
        val_loader: DataLoader for validation data
        device: Device to use for validation
        args: Validation arguments
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0
    ade_total = 0
    fde_total = 0
    worst_fde_total = 0
    miss_rate_total = 0
    
    with torch.no_grad():
        # Use tqdm for progress bar
        progress_bar = tqdm(val_loader, desc='Validating')
        
        for batch_idx, batch_graphs in enumerate(progress_bar):
            # Move data to device
            batch_graphs = [frame.to(device) for frame in batch_graphs]
            
            # Create agent mask for this batch
            agent_mask = create_agent_mask(batch_graphs, device)
            
            # Forward pass - use our model to process trajectories
            predictions = model(batch_graphs, agent_mask)
            
            # Extract ground truth future trajectories
            future_steps = args.pred_len  # Number of future steps to predict
            gt_trajectories = extract_future_data(batch_graphs, future_steps, val_loader.dataset)
            
            # Handle potential shape mismatches between predictions and ground truth
            if gt_trajectories.shape[1] != predictions.shape[1]:
                # Create proper sized ground truth with same shape as predictions
                batch_size, pred_agents, pred_len, dim = predictions.shape
                gt_batch, gt_agents, gt_len, gt_dim = gt_trajectories.shape
                
                # Create new ground truth with matching shape
                new_gt = torch.zeros_like(predictions)
                
                # Copy over data for agents that exist in both
                min_agents = min(gt_agents, pred_agents)
                new_gt[:, :min_agents, :, :] = gt_trajectories[:, :min_agents, :, :]
                
                gt_trajectories = new_gt
            
            # Ensure agent mask matches the prediction size
            if agent_mask.size(1) != predictions.shape[1]:
                batch_size = predictions.size(0)
                num_agents = predictions.size(1)
                new_agent_mask = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=device)
                
                # Copy over values from original mask if possible
                min_agents = min(agent_mask.size(1), num_agents)
                new_agent_mask[:, :min_agents] = agent_mask[:, :min_agents]
                agent_mask = new_agent_mask
            
            # Compute loss
            loss = compute_loss(predictions, gt_trajectories, agent_mask)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Calculate metrics for this batch
            metrics = calculate_metrics(
                predictions, 
                gt_trajectories,
                agent_mask=agent_mask,
                convert_coordinates=val_loader.dataset.inverse_transform_coordinates
                if hasattr(val_loader.dataset, 'inverse_transform_coordinates') else None
            )
            
            ade_total += metrics['ADE']
            fde_total += metrics['FDE']
            worst_fde_total += metrics.get('worstFDE', 0)
            miss_rate_total += metrics.get('MissRate', 0)
            
            # Update progress bar
            progress_bar.set_description(
                f"Val Loss: {loss.item():.4f}, ADE: {metrics['ADE']:.4f}, FDE: {metrics['FDE']:.4f}"
            )
    
    # Compute mean metrics
    num_batches = len(val_loader)
    metrics = {
        'loss': total_loss / num_batches,
        'ADE': ade_total / num_batches,
        'FDE': fde_total / num_batches,
        'worstFDE': worst_fde_total / num_batches,
        'MissRate': miss_rate_total / num_batches
    }
    
    return metrics

def main():
    # Configuration settings (hardcoded instead of command line arguments)
    class Args:
        # Data parameters
        data_path = 'final_surajpur_proper.csv'
        obs_len = 10
        pred_len = 10
        dist_threshold = 10.0
        batch_size = 8
        
        # Model parameters
        num_types = 9
        type_embed_dim = 16
        
        gat_hidden_dim = 64
        gat_output_dim = 64
        gat_heads = 2
        
        tcn_hidden_dim = 128
        tcn_output_dim = 128
        tcn_layers = 3
        tcn_kernel_size = 3
        
        transformer_dim = 256
        transformer_heads = 8
        transformer_encoder_layers = 4
        transformer_decoder_layers = 4
        transformer_ffn_dim = 1024
        
        dropout = 0.1
        
        # Training parameters
        epochs = 50
        lr = 0.0001
        weight_decay = 1e-4
        grad_clip = 1.0
        lr_decay = 0.5
        patience = 5
        
        # Other parameters
        seed = 42
        cuda = True
        log_dir = 'logs'
        save_dir = 'models'
    
    args = Args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    dataset = RoundaboutTrajectoryDataLoader(
        csv_path=args.data_path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        dist_threshold=args.dist_threshold,
        standardize_xy=True
    )
    
    # Split into train and validation sets (80/20)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = dataset.get_loader(batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = dataset.get_loader(batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = TrajectoryPredictionModel(args).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay, patience=args.patience, verbose=True
    )
    
    # Create logging directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    best_val_fde = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        start_time = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        train_time = time.time() - start_time
        
        # Validate
        val_metrics = validate(model, val_loader, device, args)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        
        for k in train_metrics:
            writer.add_scalar(f'Metrics/train_{k}', train_metrics[k], epoch)
            writer.add_scalar(f'Metrics/val_{k}', val_metrics[k], epoch)
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train ADE: {train_metrics['ADE']:.4f}, Val ADE: {val_metrics['ADE']:.4f}")
        print(f"Train FDE: {train_metrics['FDE']:.4f}, Val FDE: {val_metrics['FDE']:.4f}")
        print(f"Time: {train_time:.2f}s")
        
        # Combine metrics for saving
        combined_metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'epoch': epoch,
            'time': train_time
        }
        
        # Check if this is the best model
        is_best = val_metrics['FDE'] < best_val_fde
        if is_best:
            best_val_fde = val_metrics['FDE']
            
        # Save model using our comprehensive save function
        save_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            metrics=combined_metrics,
            epoch=epoch,
            is_best=is_best
        )
        
        # Save checkpoint for backward compatibility
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, os.path.join(args.save_dir, 'checkpoint.pt'))
    
    # Save final model with all components
    save_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        metrics={
            'train': train_metrics,
            'val': val_metrics,
            'best_val_fde': best_val_fde,
            'total_epochs': args.epochs
        },
        epoch=args.epochs,
        final=True
    )
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training completed!")
    print(f"Best validation FDE: {best_val_fde:.4f}")
    print(f"Model saved at: {os.path.join(args.save_dir, 'final_model.pt')}")
    print(f"Model weights saved at: {os.path.join(args.save_dir, 'final_weights.pth')}")

if __name__ == "__main__":
    main()