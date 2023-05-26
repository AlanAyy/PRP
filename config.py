params = dict()

# Data parameters
params["exp_name"] = "default"
params["dataset_type"] = "UCF-101"
params["dataset_path"] = "D:/Projects/ai-research-school/datasets/UCF101"
params["dataset_split"] = "1"
params["num_classes"] = 101  # 101
params["save_path_base"] = "D:/Projects/ai-research-school/PRP-video-pace/PRP/outputs/"
# params["pretrained_weights_path"] = "D:/Projects/ai-research-school/PRP-video-pace/PRP/outputs/pretrained/c3d.pth.tar"
params["pretrained_weights_path"] = "D:/Projects/ai-research-school/PRP-video-pace/PRP/outputs/ft_classify_default_UCF-101/05-25-18-45-three-classes-LRvideopace/best_loss_model_14.pth.tar"

# Training parameters
params["epoch_num"] = 300  # 300
params["batch_size"] = 8  # 8
params["step"] = 10  # 10

# Model parameters
params["model_type"] = "c3d"  # c3d
params["device"] = "best"  # best

# Hyperparameters
params["learning_rate"] = 0.001  # 0.001
params["momentum"] = 0.9  # 0.9
params["weight_decay"] = 0.0005  # 0.0005

# Miscellaneous
params["gpu"] = '0' # '0, 1, 2, 3'
params["num_workers"] = 4  # 4
params["print_freq"] = 500  # 500
params["save_freq"] = 10  # 10
