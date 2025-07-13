@echo off

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 128 --K 1 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k1b128.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 128 --K 2 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k2b128.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 128 --K 4 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k4b128.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 128 --K 8 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k8b128.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 32 --K 2 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k2b32.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 32 --K 4 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k4b32.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 32 --K 8 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k8b32.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 32 --K 16 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k16b32.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 32 --K 32 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k32b32.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 32 --K 64 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k64b32.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 32 --K 128 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k128b32.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 16 --K 1  --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k1b16.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 16 --K 2  --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k2b16.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 16 --K 4 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k4b16.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 16 --K 8 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k8b16.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 16 --K 16 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k16b16.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 16 --K 32 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k32b16.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 16 --K 64 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k64b16.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 16 --K 128 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k128b16.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 8 --K 1  --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k1b8.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 8 --K 2  --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k2b8.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 8 --K 4 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k4b8.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 8 --K 8 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k8b8.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 8 --K 16 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k16b8.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 8 --K 32 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k32b8.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 8 --K 64 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k64b8.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 8 --K 128 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k128b8.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 4 --K 1  --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k1b4.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 4 --K 2  --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k2b4.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 4 --K 4 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k4b4.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 4 --K 8 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k8b4.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 4 --K 16 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k16b4.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 4 --K 32 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k32b4.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 4 --K 64 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k64b4.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 4 --K 128 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k128b4.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 2 --K 1  --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k1b2.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 2 --K 2  --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k2b2.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 2 --K 4 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k4b2.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 2 --K 8 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k8b2.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 2 --K 16 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k16b2.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 2 --K 32 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k32b2.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 2 --K 64 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k64b2.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 2 --K 128 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k128b2.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 1 --K 1  --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k1b1.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 1 --K 2  --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k2b1.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 1 --K 4 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k46b1.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 1 --K 8 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k86b1.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 1 --K 16 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k16b1.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 1 --K 32 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k32b1.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 1 --K 64 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k64b1.pth

python train_pf_batchVer.py --train_ply_folder ../dataset/test --val_ply_folder ../dataset/val --num_bins 1 --K 128 --epochs 150 --lr 1e-3 --save_path pth/mppn_pf_k128b1.pth