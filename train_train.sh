#! bash

python train_train.py --preprocess False --save_path logs/search_exss_v1 --data_path datasets/Rain200H --gpu_id 0,1,2,3 --batch_size 16
