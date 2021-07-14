#! bash

# Rain100H
#python train_PReNet.py --preprocess True --save_path logs/dunetv4_128_aug_v1 --data_path datasets/train/RainTrainH --gpu_id 6 --batch_size 16
#python train_PReNet.py --preprocess True --save_path logs/dunet_128_aug_v1 --data_path datasets/train/RainTrain200H --gpu_id 6 --batch_size 16
#python train_PReNet.py --preprocess True --save_path logs/rain_dataset_prnet_100_v1 --data_path /ssd1/ruijie/rain_dataset/train --gpu_id 0 --batch_size 32
python train_PReNet_search.py --preprocess False --save_path logs/search_128_exss_v1 --data_path datasets/train/RainTrainH --gpu_id 0,1,6,7 --batch_size 4
#python train_PReNet_train.py --preprocess True --save_path logs/search_128_TEST_V9_RainDS --data_path /ssd1/ruijie/RS_RD_1000/train --gpu_id 6,7 --batch_size 24 --arch TEST_V9
#python train_PReNet_train.py --preprocess True --save_path logs/search_TEST_V9_RainDrop --data_path datasets/train/DerainDrop --gpu_id 1,2 --batch_size 24 --arch TEST_V9
#python train_PReNet_train.py --preprocess True --save_path logs/202107_search_TEST_V9_Rain200H --data_path datasets/train/RainTrain200H/ --gpu_id 4,5 --batch_size 24 --arch TEST_V9
#python train_PReNet_train.py --preprocess True --save_path logs/search_TEST_V9_Rain1200 --data_path /ssd1/ruijie/Rain1200/train/ --gpu_id 6,7 --batch_size 24 --arch TEST_V9
#
#python train_PReNet_train_co_v2.py --preprocess True --save_path logs/noshare_search_128_CO_NNV2 --data_path /ssd1/ruijie/RainDS_real_small/train_set/ --gpu_id 6,7 --batch_size 12 --rd_arch RD_V2 --rs_arch TRAIN_V1
#python train_PReNet_train_cov2_nosuper.py --save_path logs/nosupervision_search_128_CO_NNV2 --data_path /ssd1/ruijie/RainDS_real_small/train_set/ --gpu_id 1,2 --batch_size 12 --rd_arch RD_V2 --rs_arch TRAIN_V1
#python train_PReNet_train_co_v2.py --preprocess True --save_path logs/202107_new_real_search_128_CO_NNV2 --data_path /ssd1/ruijie/RainDS_real_small/train_set --gpu_id 0,1 --batch_size 12 --rd_arch RD_V2 --rs_arch TRAIN_V1





#python train_PReNet_train.py --preprocess True --save_path logs/128_dunet_DerainDrop_v1 --data_path datasets/train/DerainDrop --gpu_id 1 --batch_size 16
#python train_PReNet.py --preprocess True --save_path logs/RainDropL --data_path datasets/train/RainDropL
#python train_PReNet.py --preprocess False --save_path logs/RainDrop_Streak --data_path datasets/train/ours


# Rain100L
#python train_PReNet.py --preprocess True --save_path logs/Rain100L/PReNet --data_path datasets/train/RainTrainL

# Rain12600
#python train_PReNet.py --preprocess True --save_path logs/Rain1400/PReNet --data_path datasets/train/Rain12600
