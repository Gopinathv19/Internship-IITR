import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin/cos functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model] when batch_first=True
            
        Returns:
            Tensor with positional encoding added
        """
        # Add positional encoding to the seq_len dimension
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


class TrajectoryTransformer(nn.Module):
    """
    Transformer model for trajectory prediction with handling of variable agent counts
    """
    def __init__(
        self,
        input_dim=128,        # Input dimension from TCN
        d_model=256,          # Transformer model dimension
        nhead=8,              # Number of attention heads
        num_encoder_layers=4, # Number of encoder layers
        num_decoder_layers=4, # Number of decoder layers
        dim_feedforward=1024, # Dimension of feedforward network
        dropout=0.1,          # Dropout rate
        output_dim=2,         # Output dimension (x,y coordinates)
        max_seq_len=100,      # Maximum sequence length
        pred_len=10           # Number of timesteps to predict
    ):
        super(TrajectoryTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_len = pred_len
        
        # Input projection to transformer dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Changed to batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Changed to batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Query embedding for decoder (learnable)
        self.query_embed = nn.Parameter(torch.randn(pred_len, d_model))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform distribution"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize output projection with smaller weights
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.01)
        nn.init.constant_(self.output_projection.bias, 0)
        
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1)  # Changed to match batch_first format
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, src_mask=None, src_padding_mask=None):
        """
        Forward pass of transformer
        
        Args:
            src: Output from TCN [batch_size, num_agents, seq_len, input_dim]
            src_mask: Mask for source sequence [seq_len, seq_len]
            src_padding_mask: Mask for padding [batch_size, num_agents, seq_len]
            
        Returns:
            Predicted trajectories [batch_size, num_agents, pred_len, output_dim]
        """
        batch_size, num_agents, seq_len, _ = src.shape
        device = src.device
        
        print(f"TRNS forward - Input shape: {src.shape}")
        
        # Reshape to process all agents in batch dimension
        src = src.reshape(batch_size * num_agents, seq_len, self.input_dim)
        print(f"TRNS forward - After reshape: {src.shape}")
        
        # Project input to d_model dimension
        src = self.input_projection(src)  # [batch_size*num_agents, seq_len, d_model]
        
        # No need to transpose with batch_first=True
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Process padding mask if provided
        if src_padding_mask is not None:
            print(f"TRNS forward - Padding mask original shape: {src_padding_mask.shape}")
            if src_padding_mask.size(1) * seq_len != src.size(0) * seq_len:
                print(f"TRNS forward - WARNING: Padding mask size mismatch. Adjusting mask.")
                # Create a default mask that doesn't mask anything
                src_padding_mask = torch.zeros(batch_size * num_agents, seq_len, dtype=torch.bool, device=device)
            else:
                src_padding_mask = src_padding_mask.reshape(batch_size * num_agents, seq_len)
            print(f"TRNS forward - Padding mask adjusted shape: {src_padding_mask.shape}")
        
        # Transformer encoder
        memory = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)
        print(f"TRNS forward - After encoder: {memory.shape}")
        
        # Prepare query embeddings for decoder - adjust for batch_first
        query = self.query_embed.unsqueeze(0).repeat(batch_size * num_agents, 1, 1)
        print(f"TRNS forward - Query shape: {query.shape}")
        
        # Generate causal mask for decoder
        tgt_mask = self._generate_square_subsequent_mask(self.pred_len).to(device)
        
        # Transformer decoder
        output = self.transformer_decoder(
            query, memory, tgt_mask=tgt_mask
        )
        print(f"TRNS forward - After decoder: {output.shape}")
        
        # Project to output dimension
        output = self.output_projection(output)  # [batch_size*num_agents, pred_len, output_dim]
        
        # Reshape back to [batch_size, num_agents, pred_len, output_dim]
        output = output.reshape(batch_size, num_agents, self.pred_len, self.output_dim)
        print(f"TRNS forward - Final output: {output.shape}")
        
        return output
    
    def forward_with_agent_mask(self, src, agent_mask=None):
        """
        Alternative forward method with explicit agent masking
        
        Args:
            src: Output from TCN [batch_size, num_agents, seq_len, input_dim]
            agent_mask: Boolean mask indicating valid agents [batch_size, num_agents]
            
        Returns:
            Predicted trajectories [batch_size, num_agents, pred_len, output_dim]
        """
        batch_size, num_agents, seq_len, _ = src.shape
        device = src.device
        
        print(f"TRNS forward_with_agent_mask - Input shape: {src.shape}")
        print(f"TRNS forward_with_agent_mask - Agent mask shape: {agent_mask.shape if agent_mask is not None else None}")
        
        # Create padding mask from agent mask
        if agent_mask is not None:
            # Check if the agent_mask matches the number of agents in src
            if agent_mask.size(1) != num_agents:
                print(f"TRNS WARNING: Agent mask size ({agent_mask.size(1)}) doesn't match number of agents ({num_agents}). Adjusting mask.")
                # Create a new mask of the correct size
                new_agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool, device=device)
                # Copy over the values from the original mask for the agents that exist
                min_agents = min(agent_mask.size(1), num_agents)
                new_agent_mask[:, :min_agents] = agent_mask[:, :min_agents]
                agent_mask = new_agent_mask
                print(f"TRNS - Adjusted agent mask shape: {agent_mask.shape}")
            
            # Convert to padding mask format
            src_padding_mask = ~agent_mask.unsqueeze(-1).repeat(1, 1, seq_len)
            src_padding_mask = src_padding_mask.reshape(batch_size * num_agents, seq_len)
            print(f"TRNS forward_with_agent_mask - Created padding mask: {src_padding_mask.shape}")
        else:
            src_padding_mask = None
        
        # Forward pass with padding mask
        return self.forward(src, src_padding_mask=src_padding_mask)