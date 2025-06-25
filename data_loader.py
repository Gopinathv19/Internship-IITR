import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler


def collate_graph_sequences(batch):
    transposed = list(zip(*batch))  
    batched = [Batch.from_data_list(frame_graphs) for frame_graphs in transposed]
    return batched  

class RoundaboutTrajectoryDataLoader(Dataset):
    def __init__(self, csv_path, obs_len=10, pred_len=10, dist_threshold=10.0, standardize_xy=True):
        """
        Args:
            csv_path (str): Path to trajectory CSV file
            obs_len (int): Number of observation frames (e.g. 10)
            pred_len (int): Number of prediction frames (can be ignored if just encoding)
            dist_threshold (float): Distance threshold for edge creation (in meters)
            standardize_xy (bool): Whether to standardize the x and y coordinates
        """
        self.data = pd.read_csv(csv_path)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.dist_threshold = dist_threshold
        self.standardized_dist_threshold = dist_threshold  # Default value, will be updated if standardizing

        self.type_list = sorted(self.data['Type'].unique())
        self.num_types = len(self.type_list)
        
        # Standardize x and y coordinates if requested
        self.standardize_xy = standardize_xy
        if self.standardize_xy:
            # Use scikit-learn's StandardScaler
            self.xy_scaler = StandardScaler()
            xy_columns = ['x [m]', 'y [m]']
            xy_values = self.data[xy_columns].values
            
            # Fit the scaler and transform the data
            xy_scaled = self.xy_scaler.fit_transform(xy_values)
            
            # Update the dataframe with standardized values
            self.data['x [m]'] = xy_scaled[:, 0]
            self.data['y [m]'] = xy_scaled[:, 1]
            
            # Update the distance threshold for standardized space
            # Using average of x and y scales for distance calculation
            avg_scale = (self.xy_scaler.scale_[0] + self.xy_scaler.scale_[1]) / 2
            self.standardized_dist_threshold = self.dist_threshold / avg_scale
            print(f"Original distance threshold: {self.dist_threshold} meters")
            print(f"Standardized distance threshold: {self.standardized_dist_threshold}")
        
        self.sequences = self._build_sequences()

    def _build_sequences(self):
        frames = sorted(self.data['Time'].unique())
        sequences = []
        for i in range(len(frames) - self.obs_len - self.pred_len):
            obs_frames = frames[i:i+self.obs_len]
            sequences.append(obs_frames)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns:
            A list of PyG Data objects for each observation frame (length = obs_len)
        """
        frame_sequence = self.sequences[idx]
        graph_seq = []

        for t in frame_sequence:
            frame_data = self.data[self.data['Time'] == t]
            positions = torch.tensor(frame_data[['x [m]', 'y [m]']].values, dtype=torch.float32)
            speed = torch.tensor(frame_data['Speed [km/h]'].values, dtype=torch.float32).unsqueeze(-1)
            tan_acc = torch.tensor(frame_data['Tan. Acc. [ms-2]'].values, dtype=torch.float32).unsqueeze(-1)
            lat_acc = torch.tensor(frame_data['Lat. Acc. [ms-2]'].values, dtype=torch.float32).unsqueeze(-1)
            type_ids = torch.tensor(frame_data['Type'].values, dtype=torch.long)

            x = torch.cat([positions, speed, tan_acc, lat_acc], dim=1)
            edge_index = self.build_edge_index(positions)

            data = Data(x=x, edge_index=edge_index)
            data.type_ids = type_ids  # store type IDs for embedding
            data.frame_time = torch.tensor([t])  # optional: for debug

            graph_seq.append(data)

        return graph_seq

    def build_edge_index(self, positions):
        """
        Constructs dynamic edge_index based on Euclidean distance.
        """
        edge_index = []
        num_nodes = len(positions)
        
        # Use the appropriate threshold based on whether coordinates are standardized
        threshold = self.standardized_dist_threshold if self.standardize_xy else self.dist_threshold
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = torch.norm(positions[i] - positions[j])
                    if dist <= threshold:
                        edge_index.append([i, j])
        if len(edge_index) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_index, dtype=torch.long).T

    def get_loader(self, batch_size=4, shuffle=True, num_workers=0):
        """
        Returns a PyTorch DataLoader using custom graph sequence batching.
        
        Args:
            batch_size: Number of sequences per batch
            shuffle: Whether to shuffle which sequences go into which batch.
                     Note: This only shuffles the order of sequences, NOT the temporal
                     order within each sequence. The temporal ordering is always preserved.
            num_workers: Number of subprocesses for data loading
        """
        # Using the global collate function instead of a local one
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,  # Shuffles which sequences go into which batch
            collate_fn=collate_graph_sequences,  # Using the module-level function
            num_workers=num_workers
        )
        
    def inverse_transform_coordinates(self, positions_normalized):
        """
        Convert standardized coordinates back to original scale
        
        Args:
            positions_normalized: Tensor or array of shape (..., 2) with normalized x,y coordinates
            
        Returns:
            Original scale coordinates
        """
        if not self.standardize_xy:
            return positions_normalized
            
        # Convert to numpy if it's a tensor
        is_tensor = torch.is_tensor(positions_normalized)
        if is_tensor:
            positions_np = positions_normalized.detach().cpu().numpy()
        else:
            positions_np = positions_normalized
            
        # Reshape if needed to ensure it's 2D for the scaler
        original_shape = positions_np.shape
        positions_np = positions_np.reshape(-1, 2)
        
        # Inverse transform
        positions_original = self.xy_scaler.inverse_transform(positions_np)
        
        # Reshape back to original shape
        positions_original = positions_original.reshape(original_shape)
        
        # Convert back to tensor if input was a tensor
        if is_tensor:
            return torch.tensor(positions_original, dtype=positions_normalized.dtype, 
                               device=positions_normalized.device)
        return positions_original
