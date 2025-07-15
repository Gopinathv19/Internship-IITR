import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import your model components
from data_loader import RoundaboutTrajectoryDataLoader
from train import TrajectoryPredictionModel, train_epoch, validate, save_model, create_agent_mask

def load_checkpoint_and_continue(model_path, data_path, epochs_to_train=10):
    """
    Load a saved model checkpoint and continue training
    
    Args:
        model_path: Path to the saved model file
        data_path: Path to the training data
        epochs_to_train: Number of additional epochs to train
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the saved checkpoint
    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # Explicitly allow all objects for your own models
    
    # Extract configuration
    model_config = checkpoint.get('model_config', {})
    
    # Create Args object from the saved configuration
    class Args:
        def __init__(self, config):
            # Data parameters
            self.data_path = data_path
            self.obs_len = config.get('obs_len', 10)
            self.pred_len = config.get('pred_len', 10)
            self.dist_threshold = 10.0
            self.batch_size = 8
            
            # Model parameters
            self.num_types = config.get('num_types', 9)
            self.type_embed_dim = config.get('type_embed_dim', 16)
            
            self.gat_hidden_dim = config.get('gat_hidden_dim', 64)
            self.gat_output_dim = config.get('gat_output_dim', 64)
            self.gat_heads = config.get('gat_heads', 2)
            
            self.tcn_hidden_dim = config.get('tcn_hidden_dim', 128)
            self.tcn_output_dim = config.get('tcn_output_dim', 128)
            self.tcn_layers = config.get('tcn_layers', 3)
            self.tcn_kernel_size = config.get('tcn_kernel_size', 3)
            
            self.transformer_dim = config.get('transformer_dim', 256)
            self.transformer_heads = config.get('transformer_heads', 8)
            self.transformer_encoder_layers = config.get('transformer_encoder_layers', 4)
            self.transformer_decoder_layers = config.get('transformer_decoder_layers', 4)
            self.transformer_ffn_dim = config.get('transformer_ffn_dim', 1024)
            
            self.dropout = config.get('dropout', 0.1)
            
            # Training parameters
            self.epochs = epochs_to_train
            self.lr = 0.0001
            self.weight_decay = 1e-4
            self.grad_clip = 1.0
            self.lr_decay = 0.5
            self.patience = 5
            
            # Other parameters
            self.seed = 42
            self.cuda = True
            self.log_dir = 'logs'
            self.save_dir = 'models'
    
    args = Args(model_config)
    
 
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create model using the same configuration
    print("Creating model with saved configuration...")
    model = TrajectoryPredictionModel(args).to(device)
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pre-trained weights successfully!")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state successfully!")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay, patience=args.patience, verbose=True
    )
    
    # Load scheduler state if available
    if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loaded scheduler state successfully!")
    
    # Get starting epoch
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Continuing training from epoch {start_epoch}")
    
    # Create data loaders
    print(f"Loading data from {data_path}")
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
    
    # Create logging directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Get best validation FDE from previous training
    best_val_fde = float('inf')
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if isinstance(metrics, dict) and 'val' in metrics:
            best_val_fde = metrics['val'].get('FDE', float('inf'))
            print(f"Previous best validation FDE: {best_val_fde:.4f}")
    
    # Training loop
    print(f"\nStarting training for {epochs_to_train} additional epochs...")
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\nEpoch {epoch}/{start_epoch + args.epochs - 1}")
        
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
            print(f"New best validation FDE: {best_val_fde:.4f}")
            
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
            'scheduler_state_dict': scheduler.state_dict(),
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
            'total_epochs': start_epoch + args.epochs - 1
        },
        epoch=start_epoch + args.epochs - 1,
        final=True
    )
    
    # Close TensorBoard writer
    writer.close()
    
    print("\nTraining completed!")
    print(f"Best validation FDE: {best_val_fde:.4f}")
    print(f"Model saved at: {os.path.join(args.save_dir, 'final_model.pt')}")
    print(f"Model weights saved at: {os.path.join(args.save_dir, 'final_weights.pth')}")

if __name__ == "__main__":
    # Configuration
    model_path = os.path.join('models', 'best_model.pt')
    data_path = 'final_surajpur_proper.csv'
    epochs_to_train = 10  # Number of additional epochs to train
    
    # Continue training
    load_checkpoint_and_continue(model_path, data_path, epochs_to_train) 