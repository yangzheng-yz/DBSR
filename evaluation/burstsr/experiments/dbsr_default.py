from evaluation.common_utils.network_param import NetworkParam


def main():
    network_list = []

    # Check the documentation of NetworkParam for other example use cases
    network_list.append(NetworkParam(network_path='/home/yutong/zheng/projects/dbsr_us/pretrained_networks/pretrained_burst/dbsr_burstsr_default.pth',
                                     unique_name='DBSR_burstsr'))                # Evaluate pre-trained network weights

    return network_list

