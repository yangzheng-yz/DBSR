from evaluation.common_utils.network_param import NetworkParam


def main():
    network_list = []

    # Check the documentation of NetworkParam for other example use cases
    network_list.append(NetworkParam(network_path='/mnt/data0/zheng/SyntheticDataset/training_results/checkpoints/dbsr/database_synthetic/DBSRNet_ep0100.pth.tar',
                                     unique_name='DBSR_syn'))                   # Evaluate pre-trained network weights

    return network_list

