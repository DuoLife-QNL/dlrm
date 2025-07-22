#!/usr/bin/env python3

import os
import subprocess
import shutil
from pathlib import Path


def build_sketch_library():
    """Build the C++ sketch library."""
    
    cafe_dir = Path(__file__).parent
    src_file = cafe_dir / "sketch.cpp"
    lib_file = cafe_dir / "libsketch.so"
    
    # Check if source file exists
    if not src_file.exists():
        print(f"Source file {src_file} not found. Please copy sketch.cpp to {cafe_dir}")
        return False
        
    # Compile the library
    try:
        cmd = [
            "g++",
            "-shared",
            "-fPIC", 
            "-O3",
            "-std=c++11",
            str(src_file),
            "-o", str(lib_file)
        ]
        
        print(f"Compiling: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if lib_file.exists():
            print(f"Successfully built {lib_file}")
            return True
        else:
            print("Build failed - library file not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("g++ compiler not found. Please install gcc/g++")
        return False


def copy_sketch_source():
    """Copy sketch.cpp from CAFE project to current directory."""
    
    cafe_dir = Path(__file__).parent
    
    # Try to find the original sketch.cpp
    possible_paths = [
        Path("../../../CAFE/tricks/sketch.cpp"),
        Path("../../../../CAFE/tricks/sketch.cpp"),
        Path("CAFE/tricks/sketch.cpp"),
    ]
    
    for src_path in possible_paths:
        abs_src = (cafe_dir / src_path).resolve()
        if abs_src.exists():
            dst_path = cafe_dir / "sketch.cpp"
            shutil.copy2(abs_src, dst_path)
            print(f"Copied {abs_src} to {dst_path}")
            return True
            
    print("Could not find sketch.cpp source file")
    print("Please manually copy CAFE/tricks/sketch.cpp to dlrm/dlrm_sim/cafe/")
    return False


if __name__ == "__main__":
    print("Building CAFE sketch library...")
    
    # First try to copy the source file
    if not Path(__file__).parent.joinpath("sketch.cpp").exists():
        if not copy_sketch_source():
            exit(1)
    
    # Then build the library
    if build_sketch_library():
        print("Build successful!")
    else:
        print("Build failed!")
        exit(1) 