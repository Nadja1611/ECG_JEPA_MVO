#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a6000
#SBATCH --nodelist=mp-gpu4-a6000-1
#SBATCH --job-name=block

#python linear_eval.py --ckpt_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA_Git/ECG_JEPA/downstream_tasks/epoch100_norm.pth --dataset ptbxl --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
python linear_eval_mvo.py --data_mvo /home/nadja/ECG_JEPA_Git/downstream/IBK_data/fold_IMH_4 --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_mimic/epoch30.pth --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task regression
#python linear_eval.py --ckpt_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA_Git/ECG_JEPA/downstream_tasks/epoch135_norm_leads.pth --dataset ptbxl --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
