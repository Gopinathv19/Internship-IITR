import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    """Single block of a Temporal Convolutional Network with residual connection"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through TCN block with proper residual connection handling
        
        Args:
            x: Input tensor of shape [N, C, L]
        Returns:
            Output tensor of shape [N, C', L']
        """
        # Save original for residual connection
        residual = x
        
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Downsample if needed
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # Handle residual connection size mismatch (due to dilation)
        if residual.size(2) != out.size(2):
            # Pad residual to match output size
            padding_size = out.size(2) - residual.size(2)
            if padding_size > 0:
                # Zero pad the residual from the right
                residual = F.pad(residual, (0, padding_size), "constant", 0)
            else:
                # Crop the residual from the right
                residual = residual[:, :, :out.size(2)]
            
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        return out


class TCN(nn.Module):
    """Temporal Convolutional Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim=None, num_layers=3, 
                 kernel_size=3, dropout=0.2, dilation_base=2):
        """
        Initialize TCN
        
        Args:
            input_dim: Number of input features
            hidden_dim: Size of hidden layers
            output_dim: Size of output features (if None, same as hidden_dim)
            num_layers: Number of TCN blocks
            kernel_size: Size of the convolving kernel
            dropout: Dropout rate
            dilation_base: Base for exponential dilation growth
        """
        super(TCN, self).__init__()
        
        # If output_dim not specified, use hidden_dim
        if output_dim is None:
            output_dim = hidden_dim
            
        # Create list of TCN blocks with increasing dilation
        blocks = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation  # Causal padding
            
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim
            
            blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout
                )
            )
            
        self.network = nn.Sequential(*blocks)
        
        # Final output layer if dimensions differ
        self.output_layer = nn.Conv1d(hidden_dim, output_dim, 1) if hidden_dim != output_dim else None
        
        # Layer norm for compatibility with transformer
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
               This can be the output from a GAT layer with temporal features
               
        Returns:
            Processed tensor [batch_size, seq_len, output_dim]
            Ready to be used as input to a transformer or other modules
        """
        # Convert from [batch, seq_len, features] to [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Apply TCN blocks
        x = self.network(x)
        
        # Apply output layer if needed
        if self.output_layer is not None:
            x = self.output_layer(x)
            
        # Convert back to [batch, seq_len, features]
        x = x.transpose(1, 2)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x