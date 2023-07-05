CUDA_VISIBLE_DEVICES=1 \
python evaluate_pixel-shift.py \
--ckpt_path /home/yutong/zheng/projects/DBSR/pretrained_networks/pretrained_syn/dbsr_synthetic_default.pth \
--dataset_path /mnt/samsung/zheng/downloaded_datasets/NightCity_1024x512 \
--trajectory_path /home/yutong/zheng/projects/DBSR/util_scripts/trajectory_step-8_range-4.pkl \
--save_path /mnt/samsung/zheng/downloaded_datasets/NightCity_1024x512/evaluation_results/pretrained-model_syn \
--cfg_file /home/yutong/zheng/projects/DBSR/configs/nightCity_eval.yaml \
--save_results