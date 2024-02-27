export MODELNAME=clip_model
export BATCHSIZE=64
export LR=1e-5
export DATASET=h_s_dataset
export FREEZE=False

python main.py --model_name=model_name\
               --batch_size=batch_size\
               --lr=lr\
               --dataset=h_s_dataset\
               --freeze=freeze