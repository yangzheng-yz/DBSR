CUDA_VISIBLE_DEVICES=1 \
python evaluate_pixel-shift.py \
--ckpt_path /mnt/samsung/zheng/downloaded_datasets/NightCity_1024x512/training_results/checkpoints/dbsr/model-8_synthetic/DBSRNet_ep0100.pth.tar \
--dataset_path /mnt/samsung/zheng/downloaded_datasets/NightCity_1024x512 \
--trajectory_path /home/yutong/zheng/projects/DBSR/util_scripts/trajectory_step-4_range-4.pkl \
--save_path /mnt/samsung/zheng/downloaded_datasets/NightCity_1024x512/evaluation_results/model-8_synthetic \
--cfg_file /home/yutong/zheng/projects/DBSR/configs/nightCity_eval.yaml \
--save_results