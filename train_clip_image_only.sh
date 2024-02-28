MODELNAME="simple_classifier"
BATCHSIZE=64
LR=0.00001
DATASET="h_s_dataset"
FREEZE=false
IMAGE_ONLY=true

python main.py --model_name=$MODELNAME\
               --batch_size=$BATCHSIZE\
               --lr=$LR\
               --dataset=$DATASET\
               --freeze=$FREEZE\
               --image_only=$IMAGE_ONLY