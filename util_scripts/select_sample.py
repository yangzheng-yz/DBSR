import os
import re

# 您提供的子文件夹名称
# folder_names = [
#     "dbsr_b4_baseline_40.030_0.9498", "dbsr_b4_1-4_step7_40.727_0.9547", "dbsr_b5_1-4_step7_41.208_0.9595", 
#     "dbsr_b6_1-4_step7_41.481_0.9602", "dbsr_b7_1-4_step7_41.501_0.9609", "dbsr_b8_1-4_step7_41.627_0.9606",
#     "dbsr_b9_baseline_41.764_0.9619", "dbsr_b9_1-4_step7_41.858_0.9627", "dbsr_b10_1-4_step7",
#     "dbsr_b16_baseline_42.185_0.9648"
# ]
folder_names = [
    "dbsr_b4_1-4_step7_40.727_0.9547", "dbsr_b5_1-4_step7_41.208_0.9595", 
    "dbsr_b6_1-4_step7_41.481_0.9602", "dbsr_b7_1-4_step7_41.501_0.9609", "dbsr_b8_1-4_step7_41.627_0.9606"
]

# 假设所有子文件夹都在一个叫做parent_dir的父目录下
parent_dir = '/mnt/7T/zheng/DBSR_results/loggings'

# 字典存储图片编号与PSNR和SSIM的列表
images_info = {}

# 正则表达式匹配PSNR和SSIM值
pattern = re.compile(r'Evaluated (\d+)/\d+ images of .*, its PSNR is ([\d.]+), its SSIM is ([\d.]+)')

# 从每个文件夹的metrics_record.txt中提取PSNR和SSIM
for folder in folder_names:
    file_path = os.path.join(parent_dir, folder, 'metrics_record.txt')
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    image_num, psnr, ssim = match.groups()
                    psnr = float(psnr)
                    ssim = float(ssim)
                    if image_num not in images_info:
                        images_info[image_num] = []
                    images_info[image_num].append((psnr, ssim))
    else:
        print(f"File {file_path} does not exist.")

# 检查PSNR是否按顺序递增，并记录满足条件的图片编号和对应的最后一个SSIM值
def is_strictly_increasing(numbers):
    return all(x < y for x, y in zip(numbers, numbers[1:]))

matching_images = {}
for image_num, metrics in images_info.items():
    psnrs = [psnr for psnr, ssim in metrics]
    if len(psnrs) == len(folder_names) and is_strictly_increasing(psnrs):
        matching_images[image_num] = metrics[-1][1]  # 取最后一个SSIM值

# 打印结果
for image_num, ssim in matching_images.items():
    print(f"Image {image_num}: SSIM = {ssim}")