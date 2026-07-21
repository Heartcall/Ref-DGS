#!/usr/bin/env bash
# Ref-DGS main-table reproduction command record.
# Generated from commands actually executed during the 2026-07-20 audit.
# Canonical training commands are kept commented because the official depth priors are corrupt.

set -euo pipefail

repo=/home/liuly/Surface_Reconstruction/Glossy/Ref-DGS
data_root=/data/liuly/dataset/3DGS
run_root=/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227
refdgs_python=/home/liuly/anaconda3/envs/refdgs/bin/python

cd "$repo"

# Repository audit (executed).
git rev-parse HEAD
git status --short
git submodule status --recursive
git remote -v
git branch --show-current
git log --oneline --decorate -20

# Environment audit (executed; GPU commands required approved host execution).
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
conda run -n refdgs python -c "import sys,torch,torchvision,numpy,open3d,cv2,scipy,PIL; print(sys.executable); print(sys.version); print(torch.__version__,torchvision.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count(),torch.backends.cudnn.version()); print(numpy.__version__,open3d.__version__,cv2.__version__,scipy.__version__,PIL.__version__)"
conda run -n refdgs python -c "import diff_surfel_2dgs,diff_surfel_rasterization,diff_surfel_rasterization_feature,diff_surfel_rasterization_real,fused_ssim,simple_knn,nvdiffrast; print('extensions_import_ok')"
nvcc --version
gcc --version
df -h "$repo" /data /data1

# Official prior metadata audit (executed; metadata only, no blobs downloaded).
curl -fsSL 'https://huggingface.co/api/datasets/njfan/Ref-DGS_Priors?blobs=true' | jq '{id,sha,lastModified,sibling_count:(.siblings|length),total_bytes:([.siblings[]|(.size // .lfs.size // 0)]|add)}'
curl -fsSL 'https://huggingface.co/api/datasets/njfan/Ref-DGS_Priors?blobs=true' | jq '{shiny:{files:([.siblings[]|select(.rfilename|startswith("Ref-NeRF/refnerf/"))]|length),bytes:([.siblings[]|select(.rfilename|startswith("Ref-NeRF/refnerf/"))|(.size // .lfs.size // 0)]|add)},glossy_syn:{files:([.siblings[]|select(.rfilename|startswith("Glossy/GlossySynthetic/"))]|length),bytes:([.siblings[]|select(.rfilename|startswith("Glossy/GlossySynthetic/"))|(.size // .lfs.size // 0)]|add)}}'

# Paper audit (executed; 9.6 MiB PDF only).
mkdir -p tmp/pdfs
curl -fsSL 'https://arxiv.org/pdf/2603.07664' -o tmp/pdfs/ref_dgs_v3.pdf
gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r180 -dFirstPage=7 -dLastPage=9 -sOutputFile=tmp/pdfs/ref_dgs_white_%02d.png tmp/pdfs/ref_dgs_v3.pdf

# Required prior download (executed after explicit authorization).
# The first two invocations were interrupted after a controlled throughput diagnosis;
# the third resumes the same immutable scope at lower concurrency.
conda run -n refdgs hf download njfan/Ref-DGS_Priors --repo-type dataset --revision f7a411e493f5aa621a1bd3dcf5344058cf8bfa0e --include 'Ref-NeRF/refnerf/**' --include 'Glossy/GlossySynthetic/**' --local-dir "$run_root/priors" --max-workers 2 --format agent
conda run -n refdgs hf download njfan/Ref-DGS_Priors --repo-type dataset --revision f7a411e493f5aa621a1bd3dcf5344058cf8bfa0e --include 'Ref-NeRF/refnerf/**' --include 'Glossy/GlossySynthetic/**' --local-dir "$run_root/priors" --max-workers 8 --format agent
conda run -n refdgs hf download njfan/Ref-DGS_Priors --repo-type dataset --revision f7a411e493f5aa621a1bd3dcf5344058cf8bfa0e --include 'Ref-NeRF/refnerf/**' --include 'Glossy/GlossySynthetic/**' --local-dir "$run_root/priors" --max-workers 2 --format agent
conda run -n refdgs hf download njfan/Ref-DGS_Priors --repo-type dataset --revision f7a411e493f5aa621a1bd3dcf5344058cf8bfa0e --include 'Ref-NeRF/refnerf/**' --include 'Glossy/GlossySynthetic/**' --local-dir "$run_root/priors" --max-workers 4 --format agent
/home/liuly/anaconda3/envs/refdgs/bin/hf download njfan/Ref-DGS_Priors --repo-type dataset --revision f7a411e493f5aa621a1bd3dcf5344058cf8bfa0e --include 'Ref-NeRF/refnerf/coffee/train/depth/r_0.pth' --include 'Ref-NeRF/refnerf/helmet/train/depth/r_0.pth' --include 'Ref-NeRF/refnerf/teapot/train/depth/r_0.pth' --include 'Ref-NeRF/refnerf/toaster/train/depth/r_0.pth' --local-dir "$run_root/priors" --max-workers 1 --format agent

# Prior validation (executed; full command output is in prior_validation_final.log).
# Every downloaded .pth was checked with torch.load and SHA-256 against its
# Hugging Face .metadata; every downloaded normal PNG was verified with Pillow.

# Authentic-prior smoke command (executed on idle RTX A5000 GPU 1; exit 1).
CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v "$refdgs_python" train.py -s "$run_root/staging/refnerf/ball" -m "$run_root/smoke/ball_i2" --eval --iterations 2 --run_dim 64 --save_iterations 2
# Render was not executed because training failed before iteration 1 and no checkpoint exists.
# CUDA_VISIBLE_DEVICES=1 "$refdgs_python" render.py -m "$run_root/smoke/ball_i2" --dataset shiny --iteration 2

# Canonical scene flags frozen from the official scripts (NOT EXECUTED).
# Shiny: ball=15000, car=25000, coffee=15000 with albedo_lr=0.002,
# helmet/teapot/toaster=20000; all use --eval --run_dim 64.
# Glossy: all scenes use the default 25000 iterations plus
# --eval --run_dim 64 --albedo_bias 2 --albedo_lr 0.0005.
