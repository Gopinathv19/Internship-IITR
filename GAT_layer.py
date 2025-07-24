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
        input_dim=5,         
        hidden_dim=64,        
        output_dim=None,     
        num_heads=2,          
        dropout=0.1,          
        use_type_embedding=True,  
        num_types=8,        
        type_embed_dim=16    
    ):
        super(SpatialGAT, self).__init__()
        
        
        if output_dim is None:
            output_dim = hidden_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_type_embedding = use_type_embedding
        
      
        if use_type_embedding:
            self.type_embedding = nn.Embedding(num_types, type_embed_dim)
            actual_input_dim = input_dim + type_embed_dim
        else:
            actual_input_dim = input_dim
            
      
        self.gat1 = GATConv(
            actual_input_dim, 
            hidden_dim // num_heads,   
            heads=num_heads, 
            dropout=dropout
        )
        
         
        self.gat2 = GATConv(
            hidden_dim,
            output_dim,
            heads=1,
            dropout=dropout
        )
        
         
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        if self.use_type_embedding:
            nn.init.xavier_uniform_(self.type_embedding.weight)
         
        
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
        
         
        all_outputs = []
        all_features = [] if return_features else None
        
        
        for b in range(batch_size):
            batch_outputs = []
            batch_features = [] if return_features else None
            
        
            for t in range(seq_len):
             
                data = data_seq[b][t]
                x, edge_index = data.x, data.edge_index
                
         
                if self.use_type_embedding:
                    print("[DEBUG]", data.type_ids.max().item() ,"to",data.type_ids.min().item())
                    print("[DEBUG]", data.type_ids.unique().tolist())
                    print("[DEBUG]", data.type_ids.shape)
                    
                    type_ids = data.type_ids
                    assert type_ids.min() >= 0  , "type_ids should be non-negative"
                    assert type_ids.max() < self.num_types, "type_ids should be less than num_types"
                    type_embeddings = self.type_embedding(type_ids)
                    
                    
                    x = torch.cat([x, type_embeddings], dim=1)
                
               
                h = self.gat1(x, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
                
                 
                h = self.gat2(h, edge_index)
                
                
                h = self.layer_norm(h)
                
                 
                batch_outputs.append(h)
                if return_features:
                    batch_features.append(x)
            
            
            batch_outputs = torch.stack(batch_outputs, dim=0)   
            all_outputs.append(batch_outputs)
            
            if return_features:
                batch_features = torch.stack(batch_features, dim=0)   
                all_features.append(batch_features)
        
        gat_outputs = torch.stack(all_outputs, dim=0)  
        
        if return_features:
            all_features = torch.stack(all_features, dim=0)   
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
        
        
        outputs = []
        batch_sizes = []
        
      
        for t in range(seq_len):
            batch = batch_list[t]
            batch_sizes.append(batch.num_graphs)
            
         
        consistent_batch_size = min(batch_sizes) if batch_sizes else 0
        
        
        max_nodes_across_seq = 0
        
         
        for t in range(seq_len):
            batch = batch_list[t]
            
            
            if len(batch.batch) > 0:
                try:
                    node_counts = batch.batch.bincount(minlength=consistent_batch_size)
                    max_nodes_this_batch = node_counts[:consistent_batch_size].max().item()
                    max_nodes_across_seq = max(max_nodes_across_seq, max_nodes_this_batch)
                except RuntimeError:
                    
                    if len(batch.batch) > 0 and consistent_batch_size > 0:
                        max_nodes_this_batch = len(batch.batch) // consistent_batch_size + 1
                        max_nodes_across_seq = max(max_nodes_across_seq, max_nodes_this_batch)
        
        
        if max_nodes_across_seq == 0:
            max_nodes_across_seq = 1   
        
         
        for t in range(seq_len):
            batch = batch_list[t]
            
            if len(batch.x) == 0 or consistent_batch_size == 0:
                
                placeholder = torch.zeros(
                    consistent_batch_size, max_nodes_across_seq, self.output_dim,
                    device=batch.edge_index.device if hasattr(batch, 'edge_index') else 'cpu'
                )
                outputs.append(placeholder)
                continue
                
            x, edge_index = batch.x, batch.edge_index
            
           
            if self.use_type_embedding:
                type_ids = batch.type_ids if hasattr(batch, 'type_ids') else torch.zeros(len(x), dtype=torch.long, device=x.device)
                type_embeddings = self.type_embedding(type_ids)
                
                
                x = torch.cat([x, type_embeddings], dim=1)
            
            
            h = self.gat1(x, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
            
            
            h = self.gat2(h, edge_index)
            
            
            h = self.layer_norm(h)
            
            
            try:
                node_counts = batch.batch.bincount(minlength=consistent_batch_size)
                
                node_counts = node_counts[:consistent_batch_size]
                
             
                h_length = h.size(0)
                cumsum = torch.cumsum(node_counts, dim=0)
                
                 
                split_sizes = []
                start_idx = 0
                
                for i in range(len(node_counts)):
                    if start_idx >= h_length:
                        split_sizes.append(0)
                        continue
                        
                    size = min(node_counts[i].item(), h_length - start_idx)
                    split_sizes.append(size)
                    start_idx += size
                
               
                unbatched = list(torch.split(h, split_sizes))
                
                
                while len(unbatched) < consistent_batch_size:
                     
                    unbatched.append(torch.zeros(0, self.output_dim, device=h.device))
                
            except RuntimeError as e:
                 
                unbatched = [torch.zeros(0, self.output_dim, device=h.device) for _ in range(consistent_batch_size)]
            
         
            padded = []
            
            for nodes in unbatched[:consistent_batch_size]:  
                if len(nodes) == 0:
                    
                    nodes = torch.zeros(max_nodes_across_seq, self.output_dim, device=h.device)
                elif nodes.size(0) < max_nodes_across_seq:
                    padding = torch.zeros(
                        max_nodes_across_seq - nodes.size(0), 
                        self.output_dim,
                        device=nodes.device
                    )
                    nodes = torch.cat([nodes, padding], dim=0)
                else:
                     
                    nodes = nodes[:max_nodes_across_seq]
                
                padded.append(nodes)
            
            
            stacked = torch.stack(padded, dim=0)
            outputs.append(stacked)
        
        
        gat_outputs = torch.stack(outputs, dim=1)
        return gat_outputs