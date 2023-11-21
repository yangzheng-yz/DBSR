import pickle as pkl
import os
from collections import defaultdict
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# traj_root_path = "/mnt/samsung/zheng/downloaded_datasets/zheng_ccvl21/training_log/viz_results/debug_psnetv13_dbsr_epoch45_validation"
grid_size = 16 # 3 for 1/4, 6 for 1/8, 9 for 1/12
burst_sz = 4
traj_root_path = "/mnt/7T/zheng/DBSR_results/loggings/b4_1-16_20231113"
f = open(os.path.join(traj_root_path, 'traj.pkl'), 'rb')
# f = open(os.path.join(traj_root_path, 'traj_int_length_0.08333333333333333.pkl'), 'rb')
pixel_shift_trajectories = pkl.load(f)
f.close()

f = open(os.path.join(traj_root_path, 'metrics.txt'), 'r')
lines = f.readlines()
f.close()
weights = []
for i in lines[2:]:
    match = re.search(r'improvement: (-?\d+\.\d+)', i)
    if match:
        # print(i.strip().split(' ')[0])
        improvement_value = float(match.group(1))
        improvement_value_rounded = round(improvement_value, 3)
        weights.append(improvement_value_rounded)
        print(improvement_value_rounded)

# 计算加权频次
trajectory_strs = [''.join(map(str, traj.flatten())) for traj in pixel_shift_trajectories[2:]]
counts = Counter()
assert len(trajectory_strs) == len(weights), "len(trajectory_strs)(%s) != len(weights)(%s)" % (len(trajectory_strs), len(weights))
for i, traj_str in enumerate(trajectory_strs):
    counts[traj_str] += (weights[i] - min(weights))  # 这里使用权重进行加权统计
    # counts[traj_str] += weights[i]  # 这里使用权重进行加权统计

# 排序所有轨迹
sorted_trajectories = sorted(counts.items(), key=lambda x: x[1], reverse=True)
labels, values = zip(*sorted_trajectories)

# 绘制所有轨迹的条形图并保存
plt.figure(figsize=(20, 6))
plt.bar(range(len(values)), values)
plt.xlabel('Trajectory No.')
plt.ylabel('Weighted Frequency Number')
plt.title('Count all trajectories on validation dataset')
plt.savefig(os.path.join(traj_root_path, 'all_trajectories.png'))

# 获取前10个最常见的轨迹
common_trajectories = counts.most_common(4)

# 绘制Top-5轨迹在4x4网格中并保存
fig, axs = plt.subplots(1, 4, figsize=(15, 3))
for i, (label, _) in enumerate(common_trajectories):
    
    # burst_sz = int(len(label) / 2)
    
    print("label is %s" % label)
    # traj_array = np.array([int(x) for x in label]).reshape(burst_sz, 2)
    traj_array = np.array([[0,0],[0,1],[8,14],[13,0]])
    axs[i].set_xlim(-0.5, grid_size+0.5)
    axs[i].set_ylim(-0.5, grid_size+0.5)
    axs[i].grid(True)
    axs[i].set_xticks(np.arange(0, grid_size, 1))
    axs[i].set_yticks(np.arange(0, grid_size, 1))
    for j in range(burst_sz-1):
        axs[i].arrow(traj_array[j, 0], traj_array[j, 1], 
                     traj_array[j+1, 0] - traj_array[j, 0], 
                     traj_array[j+1, 1] - traj_array[j, 1], 
                     head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    axs[i].set_title(f"Trajectory {i+1}")
plt.savefig(os.path.join(traj_root_path, 'top_5_trajectories_.png'))
