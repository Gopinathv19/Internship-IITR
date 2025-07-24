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

        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # position encoder 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model] when batch_first=True
            
        Returns:
            Tensor with positional encoding added
        """
         
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


class TrajectoryTransformer(nn.Module):
    """
    Transformer model for trajectory prediction with handling of variable agent counts
    """
    def __init__(
        self,
        input_dim=128,         
        d_model=256,           
        nhead=8,              
        num_encoder_layers=4, 
        num_decoder_layers=4,  
        dim_feedforward=1024,  
        dropout=0.1,           
        output_dim=2,          
        max_seq_len=100,       
        pred_len=10           
    ):
        super(TrajectoryTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_len = pred_len
        
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True   
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True   
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
         
        self.output_projection = nn.Linear(d_model, output_dim)
        
         
        self.query_embed = nn.Parameter(torch.randn(pred_len, d_model))
        
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform distribution"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
         
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.01)
        nn.init.constant_(self.output_projection.bias, 0)
        
    def _generate_square_subsequent_mask(self, sz):
       
        mask = (torch.triu(torch.ones(sz, sz)) == 1)   
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
         
        batch_size, num_agents, seq_len, _ = src.shape
        device = src.device
        
 
        src = src.reshape(-1, seq_len, self.input_dim)
        src = self.input_projection(src)   
        
       
        src = self.pos_encoder(src)
         
        memory = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)
        
     
        query = self.query_embed.unsqueeze(0).repeat(batch_size * num_agents, 1, 1)
        
      
        tgt_mask = self._generate_square_subsequent_mask(self.pred_len).to(device)
        
       
        output = self.transformer_decoder(
            query, memory, tgt_mask=tgt_mask
        )
        
        
        output = self.output_projection(output)   
        
 
        output = output.reshape(batch_size, num_agents, self.pred_len, self.output_dim)
        
        return output