## Change
export LOG3DNET_DIR='/LoGG3D-Net/'

## Model
#MODEL_NAME=blip2
MODEL_NAME=git
DO_USE_SOP=True

## Train
DO_TRAIN="False"

## Eval
DO_EVAL="True"
DO_EVAL_PARTIAL="False"

## Prepreocessing
DO_DUMP_DICT_GT="False"
DO_PREPROCESS="False"

DO_PREPROCESS_ID=False

## Dataset_len
DATASET_LEN=-1


## Eval
EVAL_STEP=2
BATCH_SIZE_EVAL=1

## Train
BATCH_SIZE_TRAIN=2
NUM_TRAIN_EPOCH=10000

#LABEL_MODE="gps"
#LABEL_MODE="hilbert"
#LABEL_MODE="label"
LABEL_MODE="hierarchical"

#EXTRA_TAG="${MODEL_NAME}_${LABEL_MODE}_seq_00_80_20_contrast_quad_hierar" # 00
#EXTRA_TAG="${MODEL_NAME}_${LABEL_MODE}_seq_02_80_20_contrast_quad_hierar" # 02
#EXTRA_TAG="${MODEL_NAME}_${LABEL_MODE}_seq_05_80_20_contrast_quad_hierar" # 02
EXTRA_TAG="${MODEL_NAME}_${LABEL_MODE}_seq_06_80_20_contrast_quad_hierar" # 06
#EXTRA_TAG="${MODEL_NAME}_${LABEL_MODE}_seq_07_80_20_contrast_quadhierar" # 07
#EXTRA_TAG="${MODEL_NAME}_${LABEL_MODE}_seq_08_80_20_contrast_quad_hierar" # 08


TRAINER_CHECKPOINT="False"


EVAL_CHECKPOINT="/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/${EXTRA_TAG}/"

#eval_chkt="checkpoint-8000" # 00
#eval_chkt="checkpoint-7000" # 02
#eval_chkt="checkpoint-4900" # 05
eval_chkt="checkpoint-5000" # 06
#eval_chkt="checkpoint-2800" # 07
#eval_chkt="checkpoint-5400" # 08



## ========== Config  ========
CONFIG_NAME=config_loggnet_${LABEL_MODE}.yaml

ID_MAX_LENGTH=10


if [ "${DO_PREPROCESS}" = "True" ]; then
    DATASET_LEN=-1    
    BATCH_SIZE_TRAIN=2    
    BATCH_SIZE_EVAL=1
    NUM_TRAIN_EPOCH=1000
    EVAL_STEP=20
fi


if [ "${DO_ONLY_EVAL}" = "True" ]; then
    export TOKENIZERS_PARALLELISM=false
    #NUM_TRAIN_EPOCH=0
fi

case ${MODEL_NAME} in
    "git")
	BATCH_SIZE_TRAIN=1
	BATCH_SIZE_EVAL=1
	
	ADAM_EPSILON=1e-05
	ADAM_BETA1=0.90
	ADAM_BETA2=0.90
	LEARNING_RATE=1e-05
	SAVE_STEPS=200
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
       --save_hit_file hit_train_asuppr.txt \
       --workers 1 \
       --model_name ${MODEL_NAME} \
       --dataset_train_len ${DATASET_LEN} \
       --dataset_eval_len ${DATASET_LEN} \
       --per_device_train_batch_size ${BATCH_SIZE_TRAIN} \
       --per_device_eval_batch_size ${BATCH_SIZE_EVAL} \
       --save_to_file \
       --remove_unused_columns False \
       --output_dir ${WORK}/checkpoints/${EXTRA_TAG}  \
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
       --resume_from_checkpoint ${TRAINER_CHECKPOINT} \
       --eval_checkpoint ${EVAL_CHECKPOINT}  \
       --extra_tag ${EXTRA_TAG} \
       --eval_chkt ${eval_chkt}\
       --logging_steps 1 \
       --fix_random_seed 666 \
