import os
import yaml

# 定义YAML文件名前缀和对应的permu_nb范围
yaml_configs = {
    "b4_1-4": [0,1,2,3,4],
    "b4_1-8": [5,6,7,8,9],
    "b4_1-12": [10,11,12,13,14]
}

# 循环遍历每个split值
for split_value in ["val", "test"]:
    # 循环遍历每个yaml文件名前缀
    for yaml_name, permu_nb_range in yaml_configs.items():
        config_file = f"configs/mice_{yaml_name}.yaml"
        
        # 读取原始的yaml文件内容
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)

        # 循环遍历当前yaml文件名前缀对应的permu_nb范围
        for idx, i in enumerate(permu_nb_range, 1):
            # 修改YAML数据结构
            data['permu_nb'] = i
            data['save_path'] = f"/mnt/7T/zheng/DBSR_results/loggings/{yaml_name}_top{idx}_{split_value}set"
            data['split'] = split_value

            # 保存修改后的yaml文件内容
            with open(config_file, 'w') as f:
                yaml.safe_dump(data, f)
            print(f"!!!!!!!!!!!!!!!!!!!{idx}!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # 打印当前的参数
            print(f"Running with permu_nb: {i}, save_path: {data['save_path']}, and split: {split_value}")

            # 运行python程序
            os.system(f"python -m eval_utils.evaluate_pixel-shift_v6 {config_file}")

        # 恢复原始的yaml文件内容
        with open(config_file, 'w') as f:
            yaml.safe_dump(data, f)
