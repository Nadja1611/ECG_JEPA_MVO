#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a6000
#SBATCH --nodelist=mp-gpu4-a6000-1
#SBATCH --job-name=block

#python finetuning.py --model_name ejepa_random --ckpt_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA_Git/ECG_JEPA/downstream_tasks/epoch125_norm_leads.pth --dataset ptbxl --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel

#python finetuning.py --model_name ejepa_random --ckpt_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA_Git/ECG_JEPA/downstream_tasks/epoch100_norm.pth --dataset ptbxl --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
#python finetuning_mvo.py --model_name ejepa_random --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_with_ptbxl/epoch100_cpsc_random.pth --data_mvo /scratch/nadja/IBK_data/fold_IMH_1 --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
#python finetuning_mvo.py --model_name ejepa_random --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_with_ptbxl/epoch100_cpsc_random.pth --data_mvo /scratch/nadja/IBK_data/fold_IMH_2 --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
#python finetuning_mvo.py --model_name ejepa_random --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_with_ptbxl/epoch100_cpsc_random.pth --data_mvo /scratch/nadja/IBK_data/fold_IMH_3 --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
#python finetuning_mvo.py --model_name ejepa_random --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_with_ptbxl/epoch100_cpsc_random.pth --data_mvo /scratch/nadja/IBK_data/fold_IMH_4 --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
#python finetuning_mvo.py --model_name ejepa_random --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_with_ptbxl/epoch100_cpsc_random.pth --data_mvo /scratch/nadja/IBK_data/fold_IMH_5 --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel

python finetuning_mvo.py --model_name ejepa_random --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_mimic/epoch45.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/IBK_data/fold_IMH_1 --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
python finetuning_mvo.py --model_name ejepa_random --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_mimic/epoch45.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/IBK_data/fold_IMH_2 --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
python finetuning_mvo.py --model_name ejepa_random --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_mimic/epoch45.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/IBK_data/fold_IMH_3 --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
python finetuning_mvo.py --model_name ejepa_random --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_mimic/epoch45.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/IBK_data/fold_IMH_4 --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
python finetuning_mvo.py --model_name ejepa_random --ckpt_dir /home/nadja/ECG_JEPA_Git/downstream/weights_mimic/epoch45.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/IBK_data/fold_IMH_5 --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel