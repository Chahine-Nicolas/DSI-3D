## Change
export LOG3DNET_DIR='./LoGG3D-Net/'


## Model
MODEL_NAME=git #blip2
DO_USE_SOP=True

## Train
DO_TRAIN="True"

## Eval
DO_EVAL="False"
DO_EVAL_PARTIAL="False"

## Prepreocessing
DO_DUMP_DICT_GT="False"
DO_PREPROCESS="False"
## Old

DO_PREPROCESS_ID=False

## Dataset_len
DATASET_LEN=-1


## Eval
EVAL_STEP=100
BATCH_SIZE_EVAL=64

## Train
BATCH_SIZE_TRAIN=512
NUM_TRAIN_EPOCH=10000

#LABEL_MODE="gps"
LABEL_MODE="hilbert"
#LABEL_MODE="label"
#LABEL_MODE="hierarchical"


EXTRA_TAG="${MODEL_NAME}_${LABEL_MODE}_seq_xx_hilbert"
ID_MAX_LENGTH=18

RESUME_CHECKPOINT="False"

EVAL_CHECKPOINT="./checkpoints/${EXTRA_TAG}/"

CHECKPOINT=./ckpts/gd_mae_finetune_kitti.pth 
CHECKPOINT= None

## ========== Config  ========
CONFIG_NAME=config_loggnet_${LABEL_MODE}.yaml


if [ "${DO_PREPROCESS}" = "True" ]; then
    DATASET_LEN=-1    
    BATCH_SIZE_TRAIN=2    
    BATCH_SIZE_EVAL=2
    NUM_TRAIN_EPOCH=1000
    EVAL_STEP=20
fi


if [ "${DO_ONLY_EVAL}" = "True" ]; then
    export TOKENIZERS_PARALLELISM=false
    #NUM_TRAIN_EPOCH=0
fi

case ${MODEL_NAME} in
    "git")
	BATCH_SIZE_TRAIN=256
	BATCH_SIZE_EVAL=32
	
	ADAM_EPSILON=1e-05
	ADAM_BETA1=0.90
	ADAM_BETA2=0.90
	LEARNING_RATE=5e-04
	SAVE_STEPS=100
        ;;                                                                                           
    "blip2")
	ADAM_EPSILON=1e-05
	ADAM_BETA1=0.90
	ADAM_BETA2=0.90
	LEARNING_RATE=1e-05
	BATCH_SIZE_TRAIN=64
	BATCH_SIZE_EVAL=32
	SAVE_STEPS=2000
	# ADAM_EPSILON=1e-04
	# ADAM_BETA1=0.95
	# ADAM_BETA2=0.99
	# LEARNING_RATE=1e-03
        ;;
esac   


export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'


python -m pdb main_80_20.py \
       --launcher none \
       --cfg_file ${CONFIG_NAME} \
       --save_hit_file hit_train.txt \
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
       --output_dir ${WORK}/checkpoints/${EXTRA_TAG}/  \
       --adam_epsilon=${ADAM_EPSILON} \
       --adam_beta1=${ADAM_BETA1} \
       --adam_beta2=${ADAM_BETA2} \
       --num_train_epochs ${NUM_TRAIN_EPOCH} \
       --learning_rate=${LEARNING_RATE} \
       --eval_steps ${EVAL_STEP} \
       --evaluation_strategy steps \
       --use_sop ${DO_USE_SOP} \
       --save_steps ${SAVE_STEPS} \
       --do_train ${DO_TRAIN} \
       --do_eval ${DO_EVAL} \
       --id_max_length ${ID_MAX_LENGTH} \
       --do_eval_partial ${DO_EVAL_PARTIAL} \
       --do_preprocess ${DO_PREPROCESS} \
       --do_dump_dict_gt ${DO_DUMP_DICT_GT} \
       --resume_from_checkpoint ${RESUME_CHECKPOINT} \
       --eval_checkpoint ${EVAL_CHECKPOINT}  \
       --extra_tag ${EXTRA_TAG} \
       --warmup_steps 100 \
       --lr_scheduler_type "cosine" \
       --logging_steps 1 #> out_${MODEL_NAME}.txt 2>&1
#        --launcher none \
#        --cfg_file ./config_loggnet_cross_eval.yaml \
#        --workers 1 \
