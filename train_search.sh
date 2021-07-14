#! bash

python train_search.py --preprocess False --save_path logs/search_v1 --data_path datasets/Rain200H --gpu_id 0,1,2,3 --batch_size 16
