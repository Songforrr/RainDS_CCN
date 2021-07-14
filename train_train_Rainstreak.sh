#! bash

python train_train.py --preprocess True --save_path logs/search_TEST_V9_Rain200H --data_path datasets/train/RainTrain200H/ --gpu_id 4,5 --batch_size 24 --arch TEST_V9
