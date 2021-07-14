#! bash 

python test.py --logdir logs/search_Rain200H --save_path results/  --data_path Rain200H/ --gpu_id 0 --arch TEST_V9
python predict.py --input_dir results/ --gt_dir Rain200H/gt/
