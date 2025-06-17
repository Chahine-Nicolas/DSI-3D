export LOG3DNET_DIR='/lustre/fswork/projects/rech/dki/ujo91el/code/these_place_reco/LoGG3D-Net/'

#EXTRA_TAG="train_vect_gps_push" #"train_full" #"train_vect_asuppr"
EXTRA_TAG="train_full_seq_07"
## Model
#MODEL_NAME=blip2
MODEL_NAME=git
DO_USE_SOP=True

## Dataset_len
DATASET_LEN=-1
DO_SELF_EVAL=True

## Preprocessing
DO_PREPROCESS=False #True
DO_PREPROCESS_ID=True

## Eval
DO_LOG3DNET_EVAL=True
EVAL_STEP=100
BATCH_SIZE_EVAL=64

## Train
BATCH_SIZE_TRAIN=256
NUM_TRAIN_EPOCH=10000
TRAINER_CHECKPOINT="False"


#CHECKPOINT=ckpts/gd_mae_pretrain_kitti.pth
CHECKPOINT=/gpfswork/rech/dki/ujo91el/code/dsi-pc/ckpts/gd_mae_finetune_kitti.pth 

#CONFIG_NAME=config_loggnet_label.yaml
#CONFIG_NAME=config_loggnet_hierarchical.yaml
CONFIG_NAME=config_loggnet_gps.yaml

if [ "${DO_PREPROCESS}" = "True" ]; then
    DATASET_LEN=-1    
    BATCH_SIZE_TRAIN=2    
    BATCH_SIZE_EVAL=2
    NUM_TRAIN_EPOCH=1000
    EVAL_STEP=20
fi

if [ "${DO_SELF_EVAL}" = "True" ]; then
    DATASET_LEN=-1
else
    CONFIG_NAME=config_loggnet_cross_eval.yaml
fi
    
if [ "${DO_ONLY_EVAL}" = "True" ]; then
    export TOKENIZERS_PARALLELISM=false
    #NUM_TRAIN_EPOCH=0
fi

case ${MODEL_NAME} in
    "git")
	BATCH_SIZE_TRAIN=512
	BATCH_SIZE_EVAL=64
	
	ADAM_EPSILON=1e-05
	ADAM_BETA1=0.90
	ADAM_BETA2=0.90
	LEARNING_RATE=1e-05
        ;;                                                                                           
    "blip2")
	ADAM_EPSILON=1e-05
	ADAM_BETA1=0.90
	ADAM_BETA2=0.90
	LEARNING_RATE=1e-05
	BATCH_SIZE_TRAIN=32
	BATCH_SIZE_EVAL=32
	
	# ADAM_EPSILON=1e-04
	# ADAM_BETA1=0.95
	# ADAM_BETA2=0.99
	# LEARNING_RATE=1e-03
        ;;
esac   



export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# if [ "${DO_PREPROCESS_ID}" = "True" ]; then
#     python -m pdb preprocess_datasets.py \
# 	   --launcher none \
# 	   --cfg_file ./config_loggnet_cross_eval.yaml \
# 	   --workers 1
# fi


#python -m pdb main.py \
python -m pdb main_loop.py \
       --launcher none \
       --cfg_file ${CONFIG_NAME} \
       --save_hit_file hit_train_asuppr_256.txt \
       --workers 1 \
       --model_name ${MODEL_NAME} \
       --dataset_train_len ${DATASET_LEN} \
       --dataset_eval_len ${DATASET_LEN} \
       --pretrained_model ${CHECKPOINT} \
       --max_ckpt_save_num 500 \
       --per_device_train_batch_size ${BATCH_SIZE_TRAIN} \
       --per_device_eval_batch_size ${BATCH_SIZE_EVAL} \
       --save_to_file \
       --remove_unused_columns False \
       --dataloader_pin_memory False \
       --output_dir ${WORK}/out/dsi_log_gps/  \
       --adam_epsilon=${ADAM_EPSILON} \
       --adam_beta1=${ADAM_BETA1} \
       --adam_beta2=${ADAM_BETA2} \
       --num_train_epochs ${NUM_TRAIN_EPOCH} \
       --learning_rate=${LEARNING_RATE} \
       --eval_steps ${EVAL_STEP} \
       --evaluation_strategy steps \
       --use_sop ${DO_USE_SOP} \
       --save_steps 500 \
       --do_log3dnet_eval ${DO_LOG3DNET_EVAL} \
       --resume_from_checkpoint ${TRAINER_CHECKPOINT} \
       --extra_tag ${EXTRA_TAG} \
       --logging_steps 1 #> out_${MODEL_NAME}.txt 2>&