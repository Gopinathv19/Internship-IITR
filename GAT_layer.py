import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SpatialGAT(nn.Module):
    """
    Graph Attention Network for processing spatial relationships between agents.
    Designed to work with the RoundaboutTrajectoryDataLoader and connect to TCN.
    """
    def __init__(
        self, 
        input_dim=5,         # Default 5 features: x, y, speed, tan_acc, lat_acc
        hidden_dim=64,       # Hidden dimension size
        output_dim=None,     # Output dimension (if None, uses hidden_dim)
        num_heads=2,         # Number of attention heads in first GAT layer
        dropout=0.1,         # Dropout rate
        use_type_embedding=True,  # Whether to use vehicle type embeddings
        num_types=8,         # Maximum number of vehicle types
        type_embed_dim=16    # Dimension of type embeddings
    ):
        super(SpatialGAT, self).__init__()
        
        # Set output dimension to hidden_dim if not specified
        if output_dim is None:
            output_dim = hidden_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_type_embedding = use_type_embedding
        
        # Type embedding layer (if used)
        if use_type_embedding:
            self.type_embedding = nn.Embedding(num_types, type_embed_dim)
            actual_input_dim = input_dim + type_embed_dim
        else:
            actual_input_dim = input_dim
            
        # GAT layers
        self.gat1 = GATConv(
            actual_input_dim, 
            hidden_dim // num_heads,  # Divide by num_heads as GATConv concatenates outputs
            heads=num_heads, 
            dropout=dropout
        )
        
        # Second GAT layer (outputs single representation per node)
        self.gat2 = GATConv(
            hidden_dim,
            output_dim,
            heads=1,
            dropout=dropout
        )
        
        # Normalization and regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        if self.use_type_embedding:
            nn.init.xavier_uniform_(self.type_embedding.weight)
        
        # The GAT layers already have their own initialization in PyG
        
    def forward(self, data_seq, return_features=False):
        """
        Process a sequence of graph data from RoundaboutTrajectoryDataLoader
        
        Args:
            data_seq: List of PyG Data objects [batch_size, seq_len] 
                     (output from RoundaboutTrajectoryDataLoader)
            return_features: If True, returns intermediate node features for visualization
            
        Returns:
            gat_outputs: Tensor of node embeddings [batch_size, seq_len, num_nodes, output_dim]
        """
        batch_size = len(data_seq)
        seq_len = len(data_seq[0])
        
        # Store outputs for each batch and time step
        all_outputs = []
        all_features = [] if return_features else None
        
        # Process each batch
        for b in range(batch_size):
            batch_outputs = []
            batch_features = [] if return_features else None
            
            # Process each time step in the sequence
            for t in range(seq_len):
                # Get current graph data
                data = data_seq[b][t]
                x, edge_index = data.x, data.edge_index
                
                # Get type IDs and create embeddings if used
                if self.use_type_embedding:
                    print("[DEBUG]", data.type_ids.max().item() ,"to",data.type_ids.min().item())
                    print("[DEBUG]", data.type_ids.unique().tolist())
                    print("[DEBUG]", data.type_ids.shape)
                    
                    type_ids = data.type_ids
                    assert type_ids.min() >= 0  , "type_ids should be non-negative"
                    assert type_ids.max() < self.num_types, "type_ids should be less than num_types"
                    type_embeddings = self.type_embedding(type_ids)
                    
                    # Concatenate node features with type embeddings
                    x = torch.cat([x, type_embeddings], dim=1)
                
                # Apply first GAT layer
                h = self.gat1(x, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
                
                # Apply second GAT layer
                h = self.gat2(h, edge_index)
                
                # Apply layer normalization
                h = self.layer_norm(h)
                
                # Store outputs
                batch_outputs.append(h)
                if return_features:
                    batch_features.append(x)
            
            # Stack time steps
            batch_outputs = torch.stack(batch_outputs, dim=0)  # [seq_len, num_nodes, output_dim]
            all_outputs.append(batch_outputs)
            
            if return_features:
                batch_features = torch.stack(batch_features, dim=0)  # [seq_len, num_nodes, input_dim]
                all_features.append(batch_features)
        
        # Stack batches
        gat_outputs = torch.stack(all_outputs, dim=0)  # [batch_size, seq_len, num_nodes, output_dim]
        
        if return_features:
            all_features = torch.stack(all_features, dim=0)  # [batch_size, seq_len, num_nodes, input_dim]
            return gat_outputs, all_features
        
        return gat_outputs
    
    def process_batch(self, batch_list):
        """
        Alternative interface to process data directly from the dataloader's get_loader
        
        Args:
            batch_list: List of batched graphs [seq_len] where each element is a PyG Batch
                       (direct output from dataloader's get_loader)
                       
        Returns:
            gat_outputs: Tensor of node embeddings [batch_size, seq_len, num_nodes, output_dim]
        """
        seq_len = len(batch_list)
        
        # Process each time step
        outputs = []
        batch_sizes = []
        
        # First, determine the consistent batch_size across all timesteps
        for t in range(seq_len):
            batch = batch_list[t]
            batch_sizes.append(batch.num_graphs)
            
        # Use the minimum batch size for safety
        consistent_batch_size = min(batch_sizes) if batch_sizes else 0
        
        # Track maximum number of nodes for consistent padding
        max_nodes_across_seq = 0
        
        # First pass to determine max_nodes across all timesteps
        for t in range(seq_len):
            batch = batch_list[t]
            
            # Ensure batch.batch is not empty before calling bincount
            if len(batch.batch) > 0:
                try:
                    node_counts = batch.batch.bincount(minlength=consistent_batch_size)
                    max_nodes_this_batch = node_counts[:consistent_batch_size].max().item()
                    max_nodes_across_seq = max(max_nodes_across_seq, max_nodes_this_batch)
                except RuntimeError:
                    # If bincount fails, estimate max nodes per graph
                    if len(batch.batch) > 0 and consistent_batch_size > 0:
                        max_nodes_this_batch = len(batch.batch) // consistent_batch_size + 1
                        max_nodes_across_seq = max(max_nodes_across_seq, max_nodes_this_batch)
        
        # Ensure we have a valid max_nodes value
        if max_nodes_across_seq == 0:
            max_nodes_across_seq = 1  # Fallback to minimum value
        
        # Now process each timestep with consistent dimensions
        for t in range(seq_len):
            batch = batch_list[t]
            
            # Handle empty batch case
            if len(batch.x) == 0 or consistent_batch_size == 0:
                # Create empty placeholder output
                placeholder = torch.zeros(
                    consistent_batch_size, max_nodes_across_seq, self.output_dim,
                    device=batch.edge_index.device if hasattr(batch, 'edge_index') else 'cpu'
                )
                outputs.append(placeholder)
                continue
                
            x, edge_index = batch.x, batch.edge_index
            
            # Get type IDs and create embeddings if used
            if self.use_type_embedding:
                type_ids = batch.type_ids if hasattr(batch, 'type_ids') else torch.zeros(len(x), dtype=torch.long, device=x.device)
                type_embeddings = self.type_embedding(type_ids)
                
                # Concatenate node features with type embeddings
                x = torch.cat([x, type_embeddings], dim=1)
            
            # Apply first GAT layer
            h = self.gat1(x, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
            
            # Apply second GAT layer
            h = self.gat2(h, edge_index)
            
            # Apply layer normalization
            h = self.layer_norm(h)
            
            # Safely compute node counts with minlength to ensure consistent dimensions
            try:
                node_counts = batch.batch.bincount(minlength=consistent_batch_size)
                # Limit to consistent batch size
                node_counts = node_counts[:consistent_batch_size]
                
                # Split up to the actual length of h to avoid indexing errors
                h_length = h.size(0)
                cumsum = torch.cumsum(node_counts, dim=0)
                
                # Handle edge case where cumsum might be larger than h_length
                split_sizes = []
                start_idx = 0
                
                for i in range(len(node_counts)):
                    if start_idx >= h_length:
                        split_sizes.append(0)
                        continue
                        
                    size = min(node_counts[i].item(), h_length - start_idx)
                    split_sizes.append(size)
                    start_idx += size
                
                # Split safely based on actual available data
                unbatched = list(torch.split(h, split_sizes))
                
                # Ensure we have the correct number of tensors
                while len(unbatched) < consistent_batch_size:
                    # Add empty tensors if needed
                    unbatched.append(torch.zeros(0, self.output_dim, device=h.device))
                
            except RuntimeError as e:
                # Fallback if bincount fails: create empty tensors of consistent size
                unbatched = [torch.zeros(0, self.output_dim, device=h.device) for _ in range(consistent_batch_size)]
            
            # Pad sequences to same length (max_nodes_across_seq for consistency)
            padded = []
            
            for nodes in unbatched[:consistent_batch_size]:  # Limit to consistent batch size
                if len(nodes) == 0:
                    # Handle completely empty case
                    nodes = torch.zeros(max_nodes_across_seq, self.output_dim, device=h.device)
                elif nodes.size(0) < max_nodes_across_seq:
                    padding = torch.zeros(
                        max_nodes_across_seq - nodes.size(0), 
                        self.output_dim,
                        device=nodes.device
                    )
                    nodes = torch.cat([nodes, padding], dim=0)
                else:
                    # Trim if there are more nodes than our consistent max
                    nodes = nodes[:max_nodes_across_seq]
                
                padded.append(nodes)
            
            # Stack into [batch_size, num_nodes, output_dim]
            stacked = torch.stack(padded, dim=0)
            outputs.append(stacked)
        
        # Stack time steps to get [batch_size, seq_len, num_nodes, output_dim]
        gat_outputs = torch.stack(outputs, dim=1)
        return gat_outputs