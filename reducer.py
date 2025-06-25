import pandas as pd
import argparse

def sample_csv(input_file, output_file, num_rows=2000, random=False):
    """
    Extract a subset of rows from a CSV file and save to a new CSV
    
    Args:
        input_file: Path to the original CSV file
        output_file: Path to save the reduced CSV file
        num_rows: Number of rows to extract (default: 2000)
        random: Whether to select random rows (default: False, takes first N rows)
    """
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    total_rows = len(df)
    print(f"Original file has {total_rows} rows")
    
    if num_rows >= total_rows:
        print(f"Requested rows ({num_rows}) exceeds available rows ({total_rows})")
        num_rows = total_rows
    
    if random:
        # Take random sample
        sampled_df = df.sample(n=num_rows, random_state=42)
        print(f"Randomly sampled {num_rows} rows")
    else:
        # Take first N rows
        sampled_df = df.head(num_rows)
        print(f"Extracted first {num_rows} rows")
    
    # Save to new CSV
    sampled_df.to_csv(output_file, index=False)
    print(f"Saved reduced dataset to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract subset of rows from a CSV file')
    parser.add_argument('input_file', type=str, help='Path to the original CSV file')
    parser.add_argument('--output_file', type=str, help='Path to save the reduced CSV file')
    parser.add_argument('--num_rows', type=int, default=2000, help='Number of rows to extract')
    parser.add_argument('--random', action='store_true', help='Select random rows instead of first N')
    
    args = parser.parse_args()
    
    # If output file not specified, create default name
    if args.output_file is None:
        input_parts = args.input_file.split('.')
        if len(input_parts) > 1:
            args.output_file = f"{'.'.join(input_parts[:-1])}_reduced_{args.num_rows}.{input_parts[-1]}"
        else:
            args.output_file = f"{args.input_file}_reduced_{args.num_rows}"
    
    sample_csv(args.input_file, args.output_file, args.num_rows, args.random)