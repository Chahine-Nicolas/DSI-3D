
## Dans le fichier main_80_20.py
Changer le path de WORK_PATH et work_path

# Lancer un entraînement

## Dans le fichier job_run_80_20.slurm

Définir les noms qui seront utilisés pour différencier le job des autres (en cas de lancement en parallèle)

Ici un exemple
#SBATCH --job-name=seq_22_80_20_contrast_quad_hilbert_suite_4900                          
#SBATCH --output=train_full_seq_22_80_20_contrast_quad_hilbert_suite_4900.out                                 
#SBATCH --error=train_full_seq_22_80_20_contrast_quad_hilbert_suite_4900.err     
EXTRA_TAG="${MODEL_NAME}_${LABEL_MODE}_seq_22_80_20_contrast_quad_hilbert_suite_4900"
--save_hit_file hit_train_seq_22_80_20_contrast_quad_hilbert_suite_4900.txt \

! Vérifier la valeur de LABEL_MODE !

## Dans le fichier kitti_dataset_{LABEL_MODE}.yaml

Définir la séquence entraînée / évaluée 

Ex :
SEQ: '22'

Et le datapath de toutes les séquences

ex :
DATA_PATH: '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/' # $WORKSF


## Dans le fichier modeling_git.py

Vérifier l'utilisation ou non de la quadruplet (ou triplet loss)

si 

do_use_contrast = True

do_use_contrast_quad = True

(utile que durant les entraînements)

alors

Dans le fichier job_run_80_20.slurm
#SBATCH --partition=gpu_p2

si 

do_use_contrast = False

do_use_contrast_quad = False

(utile pour les évaluations plus rapides)

alors

Dans le fichier job_run_80_20.slurm
#SBATCH --partition=gpu_p13 suffit


Puis sbach job_run_80_20.slurm
