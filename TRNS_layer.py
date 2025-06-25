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
        Forward pass of transformer (scene-by-scene or batch-first processing)
        
        Args:
            src: Output from TCN [batch_size, num_agents, seq_len, input_dim]
            src_mask: Optional mask for source sequence
            src_padding_mask: Optional mask for padding in source sequence
            
        Returns:
            Predicted trajectories [batch_size, num_agents, pred_len, output_dim]
        """
        # Get dimensions
        batch_size, num_agents, seq_len, _ = src.shape
        device = src.device
        
        # Project input to transformer dimension
        # First reshape to combine batch and agents: [batch_size*num_agents, seq_len, input_dim]
        src = src.reshape(-1, seq_len, self.input_dim)
        src = self.input_projection(src)  # [batch_size*num_agents, seq_len, d_model]
        
        # Apply positional encoding
        src = self.pos_encoder(src)
        
        # Apply transformer encoder
        memory = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)
        
        # Prepare query embeddings for decoder
        query = self.query_embed.unsqueeze(0).repeat(batch_size * num_agents, 1, 1)
        
        # Generate causal mask for decoder
        tgt_mask = self._generate_square_subsequent_mask(self.pred_len).to(device)
        
        # Apply transformer decoder
        output = self.transformer_decoder(
            query, memory, tgt_mask=tgt_mask
        )
        
        # Project to output dimension
        output = self.output_projection(output)  # [batch_size*num_agents, pred_len, output_dim]
        
        # Reshape back to [batch_size, num_agents, pred_len, output_dim]
        output = output.reshape(batch_size, num_agents, self.pred_len, self.output_dim)
        
        return output