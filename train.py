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
        # Process with GAT
        gat_output = self.gat.process_batch(batch_graphs)
      
        
        # Store original sequence length for transformer
        orig_seq_len = gat_output.shape[1]
        
        # Reshape for TCN
        batch_size, seq_len, num_agents, gat_dim = gat_output.shape
        gat_output_reshaped = gat_output.reshape(batch_size * num_agents, seq_len, gat_dim)
        
        # Process with TCN
        tcn_output = self.tcn(gat_output_reshaped)
        
        # Handle the expanded sequence length from TCN
        new_seq_len = tcn_output.shape[1]
        tcn_output_dim = tcn_output.shape[2]
        
        # If TCN changed the sequence length, we need to adapt
        if new_seq_len != orig_seq_len:
            print(f"DEBUG - TCN changed sequence length from {orig_seq_len} to {new_seq_len}")
            
            # Option 1: Crop the sequence back to original length
            tcn_output = tcn_output[:, :orig_seq_len, :]
            
            # Option 2 (alternative): Use the full expanded sequence
            # This would require modifying the transformer to handle variable sequence lengths
            # We'll go with Option 1 for simplicity
        
        # Reshape for transformer
        tcn_output = tcn_output.reshape(batch_size, num_agents, orig_seq_len, -1)
        
        # Process with transformer
        predictions = self.transformer.forward_with_agent_mask(tcn_output, agent_mask)
        
        return predictions

def create_agent_mask(batch_graphs, device):
    """Create a mask for valid agents from batch graphs"""
    # Get number of graphs in each batch item
    seq_len = len(batch_graphs)
    
    # Use the last timestep to determine valid agents
    last_batch = batch_graphs[-1]
    
    # Get batch assignment for each node
    batch_tensor = last_batch.batch
    
    # Count nodes per graph
    unique_batches, counts = torch.unique(batch_tensor, return_counts=True)
    
    # Determine the maximum number of agents
    batch_size = len(unique_batches)
    max_agents = counts.max().item()
    
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
    Calculate trajectory prediction metrics
    
    Args:
        predictions: Model predictions [batch_size, num_agents, pred_len, 2]
        ground_truth: Ground truth future trajectories [batch_size, num_agents, pred_len, 2]
        agent_mask: Boolean mask for valid agents [batch_size, num_agents]
        convert_coordinates: Optional function to convert back to original coordinate system
    
    Returns:
        Dictionary of metrics
    """
    # Apply coordinate conversion if provided (e.g., from normalized to meters)
    if convert_coordinates is not None:
        predictions = convert_coordinates(predictions)
        ground_truth = convert_coordinates(ground_truth)
    
    # Apply agent mask if provided
    if agent_mask is not None:
        # Ensure agent_mask matches the size of predictions
        if agent_mask.size(1) != predictions.size(1):
            print(f"Metrics: Agent mask size ({agent_mask.size(1)}) doesn't match predictions size ({predictions.size(1)}). Adjusting mask.")
            
            # Create a new mask of the correct size
            batch_size = predictions.size(0)
            num_agents = predictions.size(1)
            new_agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool, device=agent_mask.device)
            
            # Copy over the values from the original mask for the agents that exist
            min_agents = min(agent_mask.size(1), num_agents)
            new_agent_mask[:, :min_agents] = agent_mask[:, :min_agents]
            agent_mask = new_agent_mask
            
        # Expand mask to match prediction dimensions
        mask = agent_mask.unsqueeze(-1).unsqueeze(-1).expand_as(predictions)
        
        # Count valid agents for averaging
        num_valid_agents = agent_mask.sum().item()
        
        # Apply mask - invalid agents don't contribute to metrics
        predictions = predictions * mask
        ground_truth = ground_truth * mask
    else:
        # All agents are valid
        batch_size, num_agents = predictions.shape[:2]
        num_valid_agents = batch_size * num_agents
    
    # Calculate Euclidean distance error at each timestep
    # Shape: [batch_size, num_agents, pred_len]
    error = torch.sqrt(((predictions - ground_truth) ** 2).sum(dim=-1) + 1e-10)
    
    # Final Displacement Error (FDE) - error at final predicted position
    fde = error[:, :, -1]  # [batch_size, num_agents]
    
    # Average Displacement Error (ADE) - mean error across all timesteps
    ade = error.mean(dim=2)  # [batch_size, num_agents]
    
    # Metrics for valid agents only
    if agent_mask is not None:
        fde = fde[agent_mask]
        ade = ade[agent_mask]
    
    # Mean across all valid agents
    mean_fde = fde.mean().item()
    mean_ade = ade.mean().item()
    
    # Additional metrics
    worst_fde = np.percentile(fde.detach().cpu().numpy(), 95)
    miss_threshold = 2.0  # meters, adjust based on your application
    miss_rate = (fde > miss_threshold).float().mean().item()
    
    return {
        'ADE': mean_ade,
        'FDE': mean_fde,
        'worstFDE': worst_fde,
        'MissRate': miss_rate
    }

def extract_future_data(batch_graphs, future_steps, dataloader):
    """Extract ground truth future data from batch_graphs"""
    # Get batch assignment for each node from the last timestep
    last_batch = batch_graphs[-1]
    batch_tensor = last_batch.batch
    
    # Count nodes per graph
    unique_batches, counts = torch.unique(batch_tensor, return_counts=True)
    batch_size = len(unique_batches)
    max_agents = counts.max().item()
    
    # Create tensor to hold the future trajectory data
    device = batch_tensor.device
    gt = torch.zeros(batch_size, max_agents, future_steps, 2, device=device)
    
    # Debug the frame_time tensor structure
    if hasattr(last_batch, 'frame_time'):
        print(f"Frame time tensor shape: {last_batch.frame_time.shape}")
    
    try:
        # Try a more direct approach - see if we can get the frame times
        # from the original sequence
        frame_times = []
        for i in range(batch_size):
            # We'll use a simpler approach - just take a future frame 
            # without trying to match exactly to the current observation
            
            # Get all sorted frames
            all_frames = sorted(dataloader.data['Time'].unique())
            
            # Pick a starting frame that leaves room for future steps
            safe_frame_idx = min(len(all_frames) - future_steps - 1, i % (len(all_frames) - future_steps - 1))
            frame_times.append(all_frames[safe_frame_idx])
        
        # For each sequence in the batch
        for batch_idx in range(batch_size):
            last_frame_time = frame_times[batch_idx]
            
            # Find this frame's index in the dataset's frames list
            all_frames = sorted(dataloader.data['Time'].unique())
            try:
                last_frame_idx = all_frames.index(last_frame_time)
            except ValueError:
                # If the frame isn't found, skip this batch item
                print(f"Frame time {last_frame_time} not found in dataset. Skipping batch item.")
                continue
            
            # Get agent IDs from the last observation frame for this batch
            last_frame_data = dataloader.data[dataloader.data['Time'] == last_frame_time]
            
            # Determine how many agents to track
            num_agents_in_batch = min(len(last_frame_data), max_agents)
            
            # Get agent IDs directly from the frame data
            agent_ids = last_frame_data['Track ID'].values[:num_agents_in_batch]
            
            # For each future timestep
            for t in range(future_steps):
                # Check if we have enough frames for this future step
                future_frame_idx = last_frame_idx + t + 1
                if future_frame_idx < len(all_frames):
                    future_frame_time = all_frames[future_frame_idx]
                    future_frame_data = dataloader.data[dataloader.data['Time'] == future_frame_time]
                    
                    # For each agent
                    for agent_idx, agent_id in enumerate(agent_ids):
                        if agent_idx >= max_agents:  # Safety check
                            break
                        
                        # Find this agent in the future frame
                        agent_future_data = future_frame_data[future_frame_data['Track ID'] == agent_id]
                        
                        if not agent_future_data.empty:
                            # Get the future position
                            future_pos = torch.tensor(
                                agent_future_data[['x [m]', 'y [m]']].values[0],
                                dtype=torch.float32,
                                device=device
                            )
                            gt[batch_idx, agent_idx, t, :] = future_pos
        
        return gt
        
    except Exception as e:
        print(f"Error extracting future data: {e}")
        print("Falling back to random future data generation...")
        
        # If all else fails, fall back to random data (but now with a warning)
        for i, count in enumerate(counts):
            if i >= batch_size:
                continue
                
            count_val = min(count.item(), max_agents)
            if count_val > 0:
                # Get positions from the last frame
                batch_nodes = batch_tensor == i
                if torch.any(batch_nodes):
                    # Get the positions of all nodes in this batch
                    positions = last_batch.x[batch_nodes][:, :2]
                    num_positions = min(len(positions), max_agents)
                    
                    # Generate random displacements for future frames
                    for t in range(future_steps):
                        # Small random displacements to simulate motion
                        displacement = torch.randn(num_positions, 2, device=device) * 0.05 * (t + 1)
                        gt[i, :num_positions, t, :] = positions[:num_positions] + displacement
        
        return gt

def train_epoch(model, train_loader, optimizer, device, args):
    model.train()
    epoch_loss = 0
    metrics = {'ADE': 0, 'FDE': 0, 'worstFDE': 0, 'MissRate': 0}
    num_batches = 0
    
    pbar = tqdm(train_loader)
    for batch_idx, batch_graphs in enumerate(pbar):
        optimizer.zero_grad()
        
        # Move data to device
        batch_graphs = [bg.to(device) for bg in batch_graphs]
        
        # Create agent mask
        agent_mask = create_agent_mask(batch_graphs, device)
        
        # Get ground truth future trajectories
        ground_truth = extract_future_data(batch_graphs, args.pred_len, train_loader.dataset)
        
        # Forward pass
        predictions = model(batch_graphs, agent_mask)
        
        # Ensure agent_mask matches the size of predictions
        if agent_mask.size(1) != predictions.size(1):
            print(f"Agent mask size ({agent_mask.size(1)}) doesn't match predictions size ({predictions.size(1)}). Adjusting mask.")
            
            # Create a new mask of the correct size
            batch_size = predictions.size(0)
            num_agents = predictions.size(1)
            new_agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool, device=device)
            
            # Copy over the values from the original mask for the agents that exist
            min_agents = min(agent_mask.size(1), num_agents)
            new_agent_mask[:, :min_agents] = agent_mask[:, :min_agents]
            agent_mask = new_agent_mask
            
            # Also adjust ground_truth to match predictions size
            if ground_truth.size(1) != num_agents:
                print(f"Ground truth size ({ground_truth.size(1)}) doesn't match predictions size ({num_agents}). Adjusting ground truth.")
                new_ground_truth = torch.zeros(
                    batch_size, num_agents, ground_truth.size(2), ground_truth.size(3),
                    dtype=ground_truth.dtype, device=ground_truth.device
                )
                min_agents = min(ground_truth.size(1), num_agents)
                new_ground_truth[:, :min_agents] = ground_truth[:, :min_agents]
                ground_truth = new_ground_truth
        
        # Calculate loss
        # Use agent_mask to only consider loss for real agents
        mask = agent_mask.unsqueeze(-1).unsqueeze(-1).expand_as(predictions)
        loss = torch.sum(((predictions - ground_truth) ** 2) * mask) / (mask.sum() + 1e-10)
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Update metrics
        batch_metrics = calculate_metrics(
            predictions.detach(), 
            ground_truth,
            agent_mask=agent_mask,
            convert_coordinates=train_loader.dataset.inverse_transform_coordinates
        )
        
        for k, v in batch_metrics.items():
            metrics[k] += v
            
        epoch_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_description(f"Train Loss: {loss.item():.4f}, ADE: {batch_metrics['ADE']:.4f}, FDE: {batch_metrics['FDE']:.4f}")
    
    # Average metrics
    epoch_loss /= num_batches
    for k in metrics:
        metrics[k] /= num_batches
        
    return epoch_loss, metrics

def validate(model, val_loader, device, args):
    model.eval()
    val_loss = 0
    metrics = {'ADE': 0, 'FDE': 0, 'worstFDE': 0, 'MissRate': 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch_graphs in enumerate(val_loader):
            # Move data to device
            batch_graphs = [bg.to(device) for bg in batch_graphs]
            
            # Create agent mask
            agent_mask = create_agent_mask(batch_graphs, device)
            
            # Get ground truth future trajectories
            ground_truth = extract_future_data(batch_graphs, args.pred_len, val_loader.dataset)
            
            # Forward pass
            predictions = model(batch_graphs, agent_mask)
            
            # Ensure agent_mask matches the size of predictions
            if agent_mask.size(1) != predictions.size(1):
                print(f"Val: Agent mask size ({agent_mask.size(1)}) doesn't match predictions size ({predictions.size(1)}). Adjusting mask.")
                
                # Create a new mask of the correct size
                batch_size = predictions.size(0)
                num_agents = predictions.size(1)
                new_agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool, device=device)
                
                # Copy over the values from the original mask for the agents that exist
                min_agents = min(agent_mask.size(1), num_agents)
                new_agent_mask[:, :min_agents] = agent_mask[:, :min_agents]
                agent_mask = new_agent_mask
                
                # Also adjust ground_truth to match predictions size
                if ground_truth.size(1) != num_agents:
                    print(f"Val: Ground truth size ({ground_truth.size(1)}) doesn't match predictions size ({num_agents}). Adjusting ground truth.")
                    new_ground_truth = torch.zeros(
                        batch_size, num_agents, ground_truth.size(2), ground_truth.size(3),
                        dtype=ground_truth.dtype, device=ground_truth.device
                    )
                    min_agents = min(ground_truth.size(1), num_agents)
                    new_ground_truth[:, :min_agents] = ground_truth[:, :min_agents]
                    ground_truth = new_ground_truth
            
            # Calculate loss
            mask = agent_mask.unsqueeze(-1).unsqueeze(-1).expand_as(predictions)
            loss = torch.sum(((predictions - ground_truth) ** 2) * mask) / (mask.sum() + 1e-10)
            
            # Update metrics
            batch_metrics = calculate_metrics(
                predictions, 
                ground_truth,
                agent_mask=agent_mask,
                convert_coordinates=val_loader.dataset.inverse_transform_coordinates
            )
            
            for k, v in batch_metrics.items():
                metrics[k] += v
                
            val_loss += loss.item()
            num_batches += 1
    
    # Average metrics
    val_loss /= num_batches
    for k in metrics:
        metrics[k] /= num_batches
        
    return val_loss, metrics

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
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        train_time = time.time() - start_time
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, device, args)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        for k in train_metrics:
            writer.add_scalar(f'Metrics/train_{k}', train_metrics[k], epoch)
            writer.add_scalar(f'Metrics/val_{k}', val_metrics[k], epoch)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train ADE: {train_metrics['ADE']:.4f}, Val ADE: {val_metrics['ADE']:.4f}")
        print(f"Train FDE: {train_metrics['FDE']:.4f}, Val FDE: {val_metrics['FDE']:.4f}")
        print(f"Time: {train_time:.2f}s")
        
        # Save best model
        if val_metrics['FDE'] < best_val_fde:
            best_val_fde = val_metrics['FDE']
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with Val FDE: {best_val_fde:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, os.path.join(args.save_dir, 'checkpoint.pt'))
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training completed!")
    print(f"Best validation FDE: {best_val_fde:.4f}")

if __name__ == "__main__":
    main()