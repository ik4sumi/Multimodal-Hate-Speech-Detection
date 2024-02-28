MODELNAME="cross_modal_attention"
BATCHSIZE=64
LR=0.00001
DATASET="hate_speech_dataset"
FREEZE=false
TEXT_ONLY=true

python main.py --model_name=$MODELNAME\
               --batch_size=$BATCHSIZE\
               --lr=$LR\
               --dataset=$DATASET\
               --freeze=$FREEZE\
               --image_only=$TEXT_ONLY