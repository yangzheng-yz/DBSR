CUDA_VISIBLE_DEVICES=1 \
python evaluate_pixel-shift_v2.py \
--ckpt_path /mnt/data0/zheng/NightCity_1024x512/training_results/checkpoints/dbsr/model-23_synthetic/DBSRNet_ep0100.pth.tar \
--dataset_path /mnt/data0/zheng/NightCity_1024x512 \
--trajectory_path /home/yutong/zheng/projects/dbsr_us/util_scripts/trajectory_step-8_range-4.pkl \
--save_path /mnt/data0/zheng/NightCity_1024x512/evaluation_results/model-23_synthetic \
--cfg_file configs/nightCity_eval.yaml \
--specify_trajetory_num 791 \
--save_results \
# --specify_image_name Chicago_0006.png

# --save_results
# --specify_trajetory_num 1 \
# --specify_image_name Chicago_0006.png