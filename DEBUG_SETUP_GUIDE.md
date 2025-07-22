# DLRM Debug Setup Guide

## Overview
This guide provides instructions for setting up remote debugging for the DLRM project using VSCode and debugpy.

## Prerequisites
- VSCode with Python extension installed
- conda environment `torchrec_exp` activated
- Current workspace: `/home/hongzheng/Codes/torchrec_based_projects` (main project root)

## File Structure
```
torchrec_based_projects/              # VSCode workspace root
├── .vscode/launch.json              # VSCode debug configuration
├── dlrm/                            # DLRM project directory
│   ├── config.sh                    # Project configuration
│   ├── dlrm_sim/
│   │   ├── dlrm_kaggle.py          # Target debugging file
│   │   └── scripts/debug_fixed.sh  # Debug launcher script
│   └── DEBUG_SETUP_GUIDE.md        # This guide
└── torchrec/                        # Custom TorchRec installation
```

## Configuration Files

### 1. VSCode Launch Configuration (`.vscode/launch.json`)
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach (DLRM)",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}/dlrm",
                    "remoteRoot": "/home/hongzheng/Codes/torchrec_based_projects/dlrm"
                }
            ],
            "cwd": "/home/hongzheng/Codes/torchrec_based_projects/dlrm",
            "justMyCode": false,
            "redirectOutput": true,
            "console": "integratedTerminal",
            "logToFile": true,
            "subProcess": true,
            "showReturnValue": true
        }
    ]
}
```

### 2. Debug Launch Script (`dlrm/dlrm_sim/scripts/debug_fixed.sh`)
```bash
#!/bin/bash
source /home/hongzheng/Codes/torchrec_based_projects/dlrm/config.sh
cd "$DLRM_WORKING_DIR"
conda activate torchrec_exp
torchx run -s local_cwd dist.ddp -j 1x1 -m debugpy -- --listen 0.0.0.0:5678 --wait-for-client dlrm_sim/dlrm_kaggle.py
```

### 3. Project Configuration (`dlrm/config.sh`)
```bash
export DLRM_WORKING_DIR="/home/hongzheng/Codes/torchrec_based_projects/dlrm"
export PYTHONPATH="${DLRM_WORKING_DIR}:${PYTHONPATH}"
```

## Usage Instructions

### Step 1: Start Debug Server
```bash
cd dlrm/dlrm_sim
./scripts/debug_fixed.sh
```

The script will:
- Load project configuration
- Change to the correct working directory
- Activate the conda environment
- Start the debug server and wait for client connection

### Step 2: Connect VSCode Debugger
1. Open VSCode with workspace at `/home/hongzheng/Codes/torchrec_based_projects`
2. Open the target file: `dlrm/dlrm_sim/dlrm_kaggle.py`
3. Set breakpoints in the code
4. Press `F5` or go to Run → Start Debugging
5. Select **"Python: Remote Attach (DLRM)"** configuration
6. The debugger will connect and execution will begin

## Key Configuration Details

### Path Mappings
- **localRoot**: `${workspaceFolder}/dlrm` (VSCode workspace dlrm subfolder)
- **remoteRoot**: `/home/hongzheng/Codes/torchrec_based_projects/dlrm` (Execution directory)

### Working Directory
- **Script execution**: `/home/hongzheng/Codes/torchrec_based_projects/dlrm`
- **Target file**: `dlrm_sim/dlrm_kaggle.py` (relative to working directory)

## Troubleshooting

### Problem: "Breakpoint in file that does not exist"
- **Cause**: Path mapping mismatch
- **Solution**: Ensure VSCode workspace is at `/home/hongzheng/Codes/torchrec_based_projects`

### Problem: "Connection refused"
- **Cause**: Debug server not running
- **Solution**: Verify debug_fixed.sh is running and listening on port 5678

### Problem: "Cannot find module"
- **Cause**: Python path or conda environment issues
- **Solution**: Verify conda environment is activated and PYTHONPATH is set correctly

### Problem: VSCode doesn't show debug configurations
- **Cause**: launch.json not in correct location
- **Solution**: Ensure launch.json is in the main workspace `.vscode/` directory, not in subfolders

## Testing the Setup

1. **Verify file paths**:
   ```bash
   ls -la /home/hongzheng/Codes/torchrec_based_projects/dlrm/dlrm_sim/dlrm_kaggle.py
   ```

2. **Test debug script**:
   ```bash
   cd dlrm/dlrm_sim
   ./scripts/debug_fixed.sh
   ```

3. **Verify VSCode configuration**:
   - Check that `.vscode/launch.json` exists in workspace root
   - Verify path mappings match your directory structure

## Notes
- The debug server will wait for client connection before starting execution
- Use `Ctrl+C` to stop the debug server if needed
- Ensure no other processes are using port 5678
- VSCode must be opened with workspace at `/home/hongzheng/Codes/torchrec_based_projects` for proper path mapping 