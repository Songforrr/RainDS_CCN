#! bash 

$RD_GTYPE=RD_V2
$RS_GTYPE=TRAIN_V1

python test_co.py --logdir logs/search_NNV1 --save_path results/search_rdrs/  --data_path RainDS_syn/rain/ --gpu_id 3 --rd_arch $RD_GTYPE --rs_arch $RS_GTYPE
python predict.py --input_dir results/search_rdrs/ --gt_dir RainDS_syn/gt/
