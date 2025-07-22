#!/usr/bin/env python3
# Author: Hongzheng-Li

import os
import sys
from typing import Tuple
import numpy as np

def count_lines(filename: str) -> int:
    """Count the number of lines in a file."""
    with open(filename, 'r') as f:
        return sum(1 for _ in f)

def split_file(input_file: str, train_ratio: float = 0.8) -> Tuple[str, str]:
    """
    Split the input file into train and validation sets.
    
    Args:
        input_file: Path to the input file
        train_ratio: Ratio of data to use for training (default: 0.8)
        
    Returns:
        Tuple of (train_file_path, val_file_path)
    """
    # Get the total number of lines
    total_lines = count_lines(input_file)
    train_lines = int(total_lines * train_ratio)
    
    # Create output file names
    dir_path = os.path.dirname(input_file)
    train_file = os.path.join(dir_path, 'train.txt')
    val_file = os.path.join(dir_path, 'val.txt')
    
    # Read and split the file
    with open(input_file, 'r') as f_in, \
         open(train_file, 'w') as f_train, \
         open(val_file, 'w') as f_val:
        
        # Write training data
        for i, line in enumerate(f_in):
            if i < train_lines:
                f_train.write(line)
            else:
                f_val.write(line)
    
    return train_file, val_file

def main():
    if len(sys.argv) != 2:
        print("Usage: python split_train_val.py <input_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
    
    print(f"Starting to split {input_file}...")
    print("This may take a while for large files...")
    
    # Backup original file
    backup_file = input_file + '.backup'
    print(f"Creating backup of original file as {backup_file}")
    os.rename(input_file, backup_file)
    
    # Split the file
    train_file, val_file = split_file(backup_file)
    
    # Print statistics
    total_lines = count_lines(backup_file)
    train_lines = count_lines(train_file)
    val_lines = count_lines(val_file)
    
    print("\nSplit completed!")
    print(f"Total lines: {total_lines}")
    print(f"Training lines: {train_lines} ({train_lines/total_lines*100:.2f}%)")
    print(f"Validation lines: {val_lines} ({val_lines/total_lines*100:.2f}%)")
    print(f"\nFiles created:")
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}")
    print(f"Original file backup: {backup_file}")

if __name__ == "__main__":
    main() 