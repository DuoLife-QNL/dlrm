# Set the current working directory to /data/workspace/Codes/torchrec/torchrec
cd ../../

raw_txt_dataset_dir="datasets/criteo_kaggle/raw_data"
output_dir="datasets/criteo_kaggle/torchrec_processed"

python -m torchrec.datasets.scripts.npy_preproc_criteo --input_dir "$raw_txt_dataset_dir" --output_dir "$output_dir" --dataset_name "criteo_kaggle"