        batch_size, seq_len, num_agents, gat_dim = gat_output.shape
        gat_output_reshaped = gat_output.reshape(batch_size * num_agents, seq_len, gat_dim)