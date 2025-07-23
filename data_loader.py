import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler


def collate_graph_sequences(batch):
    # Separate observation graphs, future trajectories, and validity masks
    obs_graphs = []
    future_trajectories = []
    valid_masks = []  # Correct: Use plural for list
    
    for obs_seq, future_seq, mask in batch:
        obs_graphs.append(obs_seq)
        future_trajectories.append(future_seq)
        valid_masks.append(mask)  # Correct: Append to valid_masks
    
    # Batch the observation graphs
    transposed = list(zip(*obs_graphs))  
    batched_graphs = [Batch.from_data_list(frame_graphs) for frame_graphs in transposed]
    
    # Find max number of agents across all batch items
    max_agents = 89
    pred_len = future_trajectories[0].size(1)
    
    # Pad each trajectory tensor and mask to have the same number of agents
    padded_trajectories = []
    padded_masks = []  # Correct: Use plural for list
    for traj, mask in zip(future_trajectories, valid_masks):  # Correct: Iterate over both lists
        num_agents = traj.size(0)
        if num_agents < max_agents:
            # Pad trajectories
            padding = torch.zeros(max_agents - num_agents, pred_len, 2, dtype=traj.dtype, device=traj.device)
            padded_traj = torch.cat([traj, padding], dim=0)
            # Pad masks
            mask_padding = torch.zeros(max_agents - num_agents, pred_len, dtype=torch.bool, device=mask.device)  # Correct: 2D shape
            padded_mask = torch.cat([mask, mask_padding], dim=0)
        else:
            padded_traj = traj[:max_agents]
            padded_mask = mask[:max_agents]
        padded_trajectories.append(padded_traj)
        padded_masks.append(padded_mask)  # Correct: Append to padded_masks
    
    # Stack the padded trajectories and masks
    batched_futures = torch.stack(padded_trajectories)
    batched_masks = torch.stack(padded_masks)
    
    return batched_graphs, batched_futures, batched_masks

class RoundaboutTrajectoryDataLoader(Dataset):
    def __init__(self, csv_path, obs_len=10, pred_len=10, dist_threshold=10.0, standardize_xy=True):

        self.data = pd.read_csv(csv_path)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.dist_threshold = dist_threshold
        self.standardized_dist_threshold = dist_threshold   
        self.type_list = sorted(self.data['Type'].unique())
        self.num_types = len(self.type_list)
        self.standardize_xy = standardize_xy
        self.num_features =5
        if self.standardize_xy:
 
            self.feature_scaler = StandardScaler()
            feature_columns = ['x [m]', 'y [m]', 'Speed [km/h]', 'Tan. Acc. [ms-2]', 'Lat. Acc. [ms-2]']
            feature_values = self.data[feature_columns].values
            feature_scaled = self.feature_scaler.fit_transform(feature_values)
            self.data[feature_columns] = feature_scaled
            avg_scale = self.feature_scaler.scale_[0:2].mean()
            self.standardized_dist_threshold = self.dist_threshold / avg_scale        
        self.sequences = self._build_sequences()

    def _build_sequences(self):
        frames = sorted(self.data['Time'].unique())
        sequences = []
        for i in range(len(frames) - self.obs_len - self.pred_len):
            obs_frames = frames[i:i+self.obs_len] # it iwll have the list like [0,0.1,0,3,2,2.3,2.9...]
            future_frames = frames[i+self.obs_len:i+self.obs_len+self.pred_len] # it has the frame like [10.2,10.3,13.4......]
            sequences.append((obs_frames, future_frames)) # finally it has like something [([],[])] # the first list consists of the observation frame and the second have the prediction frame
            
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
 
        obs_frames, future_frames = self.sequences[idx]
        # Process observation frames
        obs_graph_seq = []
        for t in obs_frames:
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
            obs_graph_seq.append(data)
        
        # Process future frames to extract ground truth trajectories
        future_trajectories,valid_mask = self._extract_future_trajectories(future_frames)
        
        return obs_graph_seq, future_trajectories,valid_mask

    def _extract_future_trajectories(self, future_frames):  # this function is responsible to get the ground truth from the csv 
        """
        Extract future trajectory data from the specified frames
        
        Args:
            future_frames: List of future frame timestamps
            
        Returns:
            Tensor of shape [num_agents, pred_len, 2] with future (x,y) coordinates
        """
        # Get all unique agents from the future frames
        future_data = self.data[self.data['Time'].isin(future_frames)]
        unique_agents = future_data['Track ID'].unique()  # Using track id as agent identifier for now
        
        # Create tensor to hold future trajectories
        future_trajectories = torch.zeros(len(unique_agents), len(future_frames), 2, dtype=torch.float32)
        valid_mask=torch.zeros(len(unique_agents),len(future_frames),dtype=torch.bool)
        
        # For each agent, extract their trajectory across future frames
        for agent_idx, agent_track_id in enumerate(unique_agents):
            agent_data = future_data[future_data['Track ID'] == agent_track_id]
            
            # Sort by time to ensure correct temporal order
            agent_data = agent_data.sort_values('Time')
            
            # Extract positions for each future frame
            for frame_idx, frame_time in enumerate(future_frames):
                frame_agent_data = agent_data[agent_data['Time'] == frame_time]
                
                if len(frame_agent_data) > 0:
                    # Take the first occurrence if multiple (should be unique)
                    x_pos = frame_agent_data['x [m]'].iloc[0]
                    y_pos = frame_agent_data['y [m]'].iloc[0]
                    future_trajectories[agent_idx, frame_idx, 0] = x_pos
                    future_trajectories[agent_idx, frame_idx, 1] = y_pos
                    valid_mask[agent_idx, frame_idx] = True
        
        return future_trajectories,valid_mask

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
        
    def inverse_transform_features(self, features_normalized):
        if not self.standardize_xy:
            return features_normalized
        is_tensor = torch.is_tensor(features_normalized)
        if is_tensor:
            features_np = features_normalized.detach().cpu().numpy()
        else:
            features_np = features_normalized
        original_shape = features_np.shape
        features_np = features_np.reshape(-1, 5)
        features_original = self.feature_scaler.inverse_transform(features_np)
        features_original = features_original.reshape(original_shape)
        if is_tensor:
            return torch.tensor(features_original, dtype=features_normalized.dtype, 
                            device=features_normalized.device)
        return features_original



 