import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import platform
from torch.utils.data import DataLoader
 

 
try:
    from data_loader import RoundaboutTrajectoryDataLoader
    from data_loader import collate_graph_sequences
    from GAT_layer import SpatialGAT
    from TCN_layer import TCN
    from TRNS_layer import TrajectoryTransformer
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}")

 
class TrajectoryPredictionModel(nn.Module):
    def __init__(self, args):
        super(TrajectoryPredictionModel, self).__init__()
        self.gat = SpatialGAT(
            input_dim=args.input_dim,
            hidden_dim=args.gat_hidden_dim,
            output_dim=args.gat_output_dim,
            num_heads=args.gat_heads,
            dropout=args.dropout,
            use_type_embedding=True,
            num_types=args.num_types,
            type_embed_dim=args.type_embed_dim
        )
        self.tcn = TCN(
            input_dim=args.gat_output_dim,
            hidden_dim=args.tcn_hidden_dim,
            output_dim=args.tcn_output_dim,
            num_layers=args.tcn_layers,
            kernel_size=args.tcn_kernel_size,
            dropout=args.dropout
        )
        self.transformer = TrajectoryTransformer(
            input_dim=args.tcn_output_dim,
            d_model=args.transformer_dim,
            nhead=args.transformer_heads,
            num_encoder_layers=args.transformer_encoder_layers,
            num_decoder_layers=args.transformer_decoder_layers,
            dim_feedforward=args.transformer_ffn_dim,
            dropout=args.dropout,
            output_dim=2,
            max_seq_len=args.obs_len,
            pred_len=args.pred_len
        )
    
    def forward(self, batch_graphs, batched_masks):
        """
        Args:
            batch_graphs: List of graph objects from dataloader
            batched_masks: Boolean mask for valid agents [batch_size, num_agents, pred_len]
        Returns:
            predictions: Trajectory predictions [batch_size, num_agents, pred_len, 2]
        """
        gat_output = self.gat.process_batch(batch_graphs)
        batch_size, seq_len, max_agents, gat_dim = gat_output.shape
        device = gat_output.device
        gat_output = torch.clamp(gat_output, -1e6, 1e6)
        
        # Use natural agent mask
        agent_mask = batched_masks.any(dim=-1)  # [batch_size, max_agents from collate, e.g., 89]
        print(f"agent_mask shape before reshape: {agent_mask.shape}")
        
        # Reshape for TCN
        reshaped_data = gat_output.permute(0, 2, 1, 3)  # [batch_size, max_agents, seq_len, gat_dim]
        print(f"reshaped data shape {reshaped_data.shape}")
        
        flat_batch_data = reshaped_data.reshape(-1, seq_len, gat_dim)
        print(f"flat data shape type and the shape {flat_batch_data.dtype} {flat_batch_data.shape}")
        max_indices = flat_batch_data.size(0) - 1  # e.g., 735 for 736 elements
        
        # Adjust flat_agent_mask to current total agents
        current_total_agents = batch_size * max_agents
        flat_agent_mask = agent_mask.reshape(-1)[:current_total_agents]
        print(f"flat_agent_mask shape: {flat_agent_mask.shape}, sum: {flat_agent_mask.sum().item()}")
        print(f"flat_agent_mask shape {flat_agent_mask.shape} {flat_agent_mask.dtype} sum : {flat_agent_mask.sum().item()}")
        
        valid_indices = torch.where(flat_agent_mask)[0]
        if (valid_indices > max_indices).any() or (valid_indices < 0).any():
            raise ValueError(f"Invalid indices: {valid_indices}, max_indices: {max_indices}")
        
        print(f"flat_batch_data : {flat_batch_data.dtype} , flat_batch_data size : {flat_batch_data.shape} flat_batch_data : {flat_batch_data.device}")
        valid_data = flat_batch_data[valid_indices]
        print(f"Valid data min/max: {valid_data.min().item()}, {valid_data.max().item()}, has NaN: {torch.isnan(valid_data).any().item()}")
        print(f"Valid data shape {valid_data.shape},valid data type {valid_data.dtype},valid data device {valid_data.device}")
        
        if len(valid_indices) > 0:
            tcn_output = self.tcn(valid_data)
            tcn_seq_len = tcn_output.shape[1]
            if tcn_seq_len != seq_len:
                if tcn_seq_len > seq_len:
                    tcn_output = tcn_output[:, -seq_len:, :]
                else:
                    padding = torch.zeros(len(valid_indices), seq_len - tcn_seq_len, tcn_output.shape[2], device=device)
                    tcn_output = torch.cat([tcn_output, padding], dim=1)
            tcn_output = torch.clamp(tcn_output, -1e6, 1e6)
            tcn_dim = tcn_output.shape[2]
            all_tcn_output = torch.zeros(batch_size * max_agents, seq_len, tcn_dim, device=device)
            all_tcn_output[valid_indices] = tcn_output
            batch_tcn_output = all_tcn_output.reshape(batch_size, max_agents, seq_len, tcn_dim)
            
            # Process with transformer
            transformer_output = self.transformer(batch_tcn_output)
            print(f"Transformer output shape: {transformer_output.shape}")
            
            # Use max_agents for predictions (match transformer_output)
            predictions = transformer_output[:, :max_agents, :, :]  # Ensure it matches max_agents
            print(f"Predictions shape before mask: {predictions.shape}")
            
            # Adjust agent_mask to match max_agents by padding if needed
            if agent_mask.size(1) < max_agents:
                padding = torch.zeros(batch_size, max_agents - agent_mask.size(1), device=device, dtype=torch.bool)
                agent_mask = torch.cat([agent_mask, padding], dim=1)
            elif agent_mask.size(1) > max_agents:
                agent_mask = agent_mask[:, :max_agents]
            
            # Expand mask to match predictions
            expanded_mask = agent_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, max_agents, seq_len, 2)
            print(f"Expanded mask shape: {expanded_mask.shape}")
            
            # Apply mask to predictions
            predictions = predictions * expanded_mask.float()
            print(f"Final predictions shape: {predictions.shape}")
        else:
            predictions = torch.zeros(batch_size, max_agents, self.transformer.pred_len, 2, device=device)
        
        return predictions
    
 

def calculate_metrics(predictions, ground_truth, batched_masks=None, convert_coordinates=None):
    """
    Calculate trajectory prediction metrics (e.g., ADE, FDE, worst FDE, Miss Rate).
    
    Args:
        predictions: Predicted trajectories [batch_size, num_agents, pred_len, 2]
        ground_truth: Ground truth trajectories [batch_size, num_agents, pred_len, 2]
        batched_masks: Boolean mask for valid agents [batch_size, num_agents, pred_len] (optional)
        convert_coordinates: Function to convert coordinates (optional)
    
    Returns:
        metrics: Dictionary containing ADE, FDE, worstFDE, and MissRate
    """
    print(f"predictions shape: {predictions.shape}, ground_truth shape: {ground_truth.shape}")
    
    # Get agent dimensions
    batch_size_pred, pred_agents, pred_len, dim = predictions.shape
    batch_size_gt, gt_agents, _, _ = ground_truth.shape
    
    # Verify batch size consistency
    if batch_size_pred != batch_size_gt:
        raise ValueError(f"Batch size mismatch: predictions {batch_size_pred}, ground_truth {batch_size_gt}")
    
    # Truncate to minimum agent count
    min_agents = min(pred_agents, gt_agents)
    predictions = predictions[:, :min_agents, :, :]
    ground_truth = ground_truth[:, :min_agents, :, :]
    if batched_masks is not None:
        batched_masks = batched_masks[:, :min_agents, :]
    
    print(f"Adjusted predictions shape: {predictions.shape}, ground_truth shape: {ground_truth.shape}")
    
    # Verify shapes match after truncation
    if predictions.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch after truncation: predictions {predictions.shape}, ground_truth {ground_truth.shape}")
    
    if convert_coordinates is not None:
        predictions = convert_coordinates(predictions)
        ground_truth = convert_coordinates(ground_truth)
    
    batch_size, num_agents, pred_len, _ = predictions.shape
    device = predictions.device
    
    if batched_masks is None:
        batched_masks = torch.ones(batch_size, num_agents, pred_len, dtype=torch.bool, device=device)
    agent_mask = batched_masks.any(dim=-1)  # [batch_size, num_agents]
    
    all_ades = []
    all_fdes = []
    for b in range(batch_size):
        scene_mask = agent_mask[b]
        valid_agents = torch.where(scene_mask)[0]
        if len(valid_agents) == 0:
            continue
        scene_pred = predictions[b, valid_agents]
        scene_gt = ground_truth[b, valid_agents]
        error = torch.sqrt(((scene_pred - scene_gt) ** 2).sum(dim=-1) + 1e-10)
        scene_fde = error[:, -1]
        scene_ade = error.mean(dim=1)
        all_ades.append(scene_ade)
        all_fdes.append(scene_fde)
    
    if all_ades and all_fdes:
        all_ades = torch.cat(all_ades)
        all_fdes = torch.cat(all_fdes)
        mean_ade = all_ades.mean().item()
        mean_fde = all_fdes.mean().item()
        worst_fde = torch.quantile(all_fdes, 0.95).item()
        miss_threshold = 2.0
        miss_rate = (all_fdes > miss_threshold).float().mean().item()
    else:
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
    if not train_loader:
        raise ValueError("train_loader is empty")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")
    
    model.train()
    total_loss = 0
    ade_total = 0
    fde_total = 0
    num_valid_batches = 0
    
    # Use tqdm for progress bar
 
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (batch_graphs, gt_trajectories, batched_masks) in enumerate(progress_bar):
        # Move data to device
        batch_graphs = [frame.to(device) for frame in batch_graphs]
        gt_trajectories = gt_trajectories.to(device)
        batched_masks = batched_masks.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_graphs, batched_masks)
        
        # Compute loss
        loss = compute_loss(predictions, gt_trajectories, batched_masks)
        
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
                batched_masks=batched_masks,
                convert_coordinates=train_loader.dataset.inverse_transform_features
                if hasattr(train_loader.dataset, 'inverse_transform_features') else None
            )
            
            # Accumulate only valid metrics
            if not np.isnan(metrics['ADE']) and not np.isnan(metrics['FDE']):
                ade_total += metrics['ADE']
                fde_total += metrics['FDE']
                num_valid_batches += 1
            
            # Update the progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_description(
                    f"Train Loss: {loss.item():.4f}, ADE: {metrics['ADE']:.4f}, FDE: {metrics['FDE']:.4f}"
                )
            else:
                progress_bar.set_description(f"Training")
    
    # Compute mean metrics
    num_batches = len(train_loader)
    if num_valid_batches == 0:
        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else float('nan'),
            'ADE': float('nan'),
            'FDE': float('nan')
        }
    else:
        metrics = {
            'loss': total_loss / num_batches,
            'ADE': ade_total / num_valid_batches,
            'FDE': fde_total / num_valid_batches
        }
    
    return metrics

def compute_loss(predictions, targets, batched_masks=None):
    """
    Custom loss function for trajectory prediction with agent masking
    
    Args:
        predictions: Model predictions [batch_size, num_agents, pred_len, 2]
        targets: Ground truth future trajectories [batch_size, num_agents, pred_len, 2]
        batched_masks: Boolean mask for valid agents [batch_size, num_agents, pred_len]
        
    Returns:
        Loss value
    """
    print(f"predictions shape: {predictions.shape}, targets shape: {targets.shape}")
    
    # Get agent dimensions
    batch_size_pred, pred_agents, pred_len, dim = predictions.shape
    batch_size_target, target_agents, _, _ = targets.shape
    
    # Verify batch size consistency
    if batch_size_pred != batch_size_target:
        raise ValueError(f"Batch size mismatch: predictions {batch_size_pred}, targets {batch_size_target}")
    
    # Truncate to minimum agent count
    min_agents = min(pred_agents, target_agents)
    predictions = predictions[:, :min_agents, :, :]
    targets = targets[:, :min_agents, :, :]
    if batched_masks is not None:
        batched_masks = batched_masks[:, :min_agents, :]
    
    print(f"Adjusted predictions shape: {predictions.shape}, targets shape: {targets.shape}")
    
    # Verify shapes match after truncation
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch after truncation: predictions {predictions.shape}, targets {targets.shape}")
    
    device = predictions.device
    
    if batched_masks is None:
        batched_masks = torch.ones(batch_size_pred, min_agents, pred_len, dtype=torch.bool, device=device)
    elif batched_masks.shape != (batch_size_pred, min_agents, pred_len):
        raise ValueError(f"batched_masks shape {batched_masks.shape} does not match expected {(batch_size_pred, min_agents, pred_len)}")
    
    # Clip predictions and targets for numerical stability
    predictions = torch.clamp(predictions, -1e6, 1e6)
    targets = torch.clamp(targets, -1e6, 1e6)
    
    # MSE loss with masking
    squared_error = torch.pow(predictions - targets, 2).sum(dim=-1)  # [batch_size, num_agents, pred_len]
    
    # Apply mask
    masked_squared_error = squared_error * batched_masks.float()
    
    # Count number of valid elements for averaging
    num_valid = batched_masks.sum().item()
    
    # If no valid elements, return 0 loss
    if num_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
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
    if not hasattr(args, 'save_dir') or not args.save_dir:
        raise ValueError("args.save_dir must be a non-empty string")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a PyTorch Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer must be a PyTorch Optimizer")
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
        raise ValueError("scheduler must be a PyTorch LRScheduler or None")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    save_dict = {
        'model_config': {
            'input_dim': args.input_dim,
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
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    
    if final:
        save_path = os.path.join(args.save_dir, 'final_model.pt')
        torch.save(save_dict, save_path)
        print(f"Saved final model to {save_path}")
    if is_best:
        save_path = os.path.join(args.save_dir, 'best_model.pt')
        torch.save(save_dict, save_path)
        print(f"Saved best model with metrics: {metrics}")
    
    save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(save_dict, save_path)
    print(f"Saved checkpoint to {save_path}")
    
    if is_best:
        weights_path = os.path.join(args.save_dir, 'best_weights.pth')
        torch.save(model.state_dict(), weights_path)
        print(f"Saved best weights to {weights_path}")
    
    if final:
        weights_path = os.path.join(args.save_dir, 'final_weights.pth')
        torch.save(model.state_dict(), weights_path)
        print(f"Saved final weights to {weights_path}")

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
     
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA device specified but not available")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_config' not in checkpoint or 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint missing required keys: 'model_config' or 'model_state_dict'")
    
    model_config = checkpoint.get('model_config', {})
    
    class Args:
        pass
    args = Args()
    for key, value in model_config.items():
        setattr(args, key, value)
    
    model = TrajectoryPredictionModel(args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
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
 
    
    if not val_loader:
        raise ValueError("val_loader is empty")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a PyTorch Module")
    
    model.eval()
    total_loss = 0
    ade_total = 0
    fde_total = 0
    worst_fde_total = 0
    miss_rate_total = 0
    num_valid_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validating')
        
        for batch_idx, (batch_graphs, gt_trajectories, batched_masks) in enumerate(progress_bar):
            batch_graphs = [frame.to(device) for frame in batch_graphs]
            gt_trajectories = gt_trajectories.to(device)
            batched_masks = batched_masks.to(device)
            
            predictions = model(batch_graphs, batched_masks)
            
            loss = compute_loss(predictions, gt_trajectories, batched_masks)
            total_loss += loss.item()
            
            metrics = calculate_metrics(
                predictions,
                gt_trajectories,
                batched_masks=batched_masks,
                convert_coordinates=val_loader.dataset.inverse_transform_features
                if hasattr(val_loader.dataset, 'inverse_transform_features') else None
            )
            
            if not np.isnan(metrics['ADE']) and not np.isnan(metrics['FDE']):
                ade_total += metrics['ADE']
                fde_total += metrics['FDE']
                worst_fde_total += metrics.get('worstFDE', 0)
                miss_rate_total += metrics.get('MissRate', 0)
                num_valid_batches += 1
            
            if batch_idx % 10 == 0:
                progress_bar.set_description(
                    f"Val Loss: {loss.item():.4f}, ADE: {metrics['ADE']:.4f}, FDE: {metrics['FDE']:.4f}"
                )
            else:
                progress_bar.set_description("Validating")
    
    num_batches = len(val_loader)
    if num_valid_batches == 0:
        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else float('nan'),
            'ADE': float('nan'),
            'FDE': float('nan'),
            'worstFDE': float('nan'),
            'MissRate': float('nan')
        }
    else:
        metrics = {
            'loss': total_loss / num_batches,
            'ADE': ade_total / num_valid_batches,
            'FDE': fde_total / num_valid_batches,
            'worstFDE': worst_fde_total / num_valid_batches,
            'MissRate': miss_rate_total / num_valid_batches
        }
    
    return metrics

def main():
 

    # Configuration settings
    class Args:
        data_path = 'data/final_surajpur_proper.csv'
        obs_len = 10
        pred_len = 10
        dist_threshold = 10.0
        batch_size = 8
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
        epochs = 50
        lr = 0.0001
        weight_decay = 1e-4
        grad_clip = 1.0
        lr_decay = 0.5
        patience = 5
        seed = 42
        cuda = True
        log_dir = 'logs'
        save_dir = 'models'
        num_workers = 0 if platform.system() == 'Windows' else 4
    
    args = Args()
    
    # Validate inputs
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"Data file {args.data_path} does not exist")
    if not args.save_dir or not args.log_dir:
        raise ValueError("save_dir and log_dir must be non-empty strings")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders with error handling
    try:
        dataset = RoundaboutTrajectoryDataLoader(
            csv_path=args.data_path,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            dist_threshold=args.dist_threshold,
            standardize_xy=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize dataset: {e}")
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    # Set input_dim dynamically based on dataset
    args.input_dim = dataset.num_features  # Assumes DataLoader provides num_features
    
    # Split into train and validation sets (80/20)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Invalid split: train_size={train_size}, val_size={val_size}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_graph_sequences,
    num_workers=args.num_workers
)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_sequences,
        num_workers=args.num_workers
    )
    
    # Create model
    model = TrajectoryPredictionModel(args).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay, patience=args.patience
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
        is_best = not np.isnan(val_metrics['FDE']) and val_metrics['FDE'] < best_val_fde
        if is_best:
            best_val_fde = val_metrics['FDE']
        
        # Save model
        save_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            metrics=combined_metrics,
            epoch=epoch,
            is_best=is_best
        )
    
    # Save final model
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