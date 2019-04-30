import argparse


def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
# super point
net_arg.add_argument(
    "--cell", type=int, default=8, help=""
    "Detector cell size")
net_arg.add_argument(
    "--conf_thresh", type=float, default=0.001, help=""
    "Detector confidence threshold")
net_arg.add_argument(
    "--nms_dist", type=int, default=1, help=""
    "nms distance")
net_arg.add_argument(
    "--window_size", type=int, default=3, help=""
    "window size of spatial soft argmax")
net_arg.add_argument(
    "--num_kp", type=int, default=500, help=""
    "number of keypoints per image")
net_arg.add_argument(
    "--temp_init", type=float, default=10, help=""
    "init temperature in spatial soft argmax")
net_arg.add_argument(
    "--pretrain_backbone", type=str, default='../../model/superpoint_v1.pth', help=""
    "whether use order aware filtering block")
# corr net
net_arg.add_argument(
    "--net_depth", type=int, default=12, help=""
    "number of layers of OANet")
net_arg.add_argument(
    "--net_channels", type=int, default=128, help=""
    "number of channels in a layer")
net_arg.add_argument(
    "--use_oaf", type=str2bool, default=True, help=""
    "whether use order aware filtering block")
net_arg.add_argument(
    "--num_clusters", type=int, default=500, help=""
    "cluster num in oanet")
net_arg.add_argument(
    "--kpt_only", type=str2bool, default=False, help=""
    "input kpt only")
net_arg.add_argument(
    "--pretrain_corrnet", type=str, default='../../model/corrnet/yfcc/ours/superpoint/model_best.pth', help=""
    "pretrained correspondence network")


# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--dataset_path", type=str, default='../../data/dataset/yfcc', help=""
    "img path")
data_arg.add_argument(
    "--data_tr", type=str, default='../../data/yfcc-train.pkl', help=""
    "name of the dataset for train")
data_arg.add_argument(
    "--data_va", type=str, default='../../data/yfcc-val.pkl', help=""
    "name of the dataset for valid")
data_arg.add_argument(
    "--data_te", type=str, default='../../data/yfcc-test.pkl', help=""
    "name of the dataset for test")
data_arg.add_argument(
    "--img_H", type=int, default=480, help=""
    "image height")
data_arg.add_argument(
    "--img_W", type=int, default=640, help=""
    "image width")
data_arg.add_argument(
    "--input_noise", type=float, default=0, help=""
    "standard deviation of guassian noise ")
data_arg.add_argument(
    "--kpt_prefix", type=str, default='sp', help=""
    "prefix for keypoints map")

# -----------------------------------------------------------------------------
# Objective
obj_arg = add_argument_group("obj")
obj_arg.add_argument(
    "--obj_top_k", type=int, default=-1, help=""
    "number of keypoints above the threshold to use for "
    "essential matrix estimation. put -1 to use all. ")
obj_arg.add_argument(
    "--obj_geod_th", type=float, default=1e-4, help=""
    "theshold for the good geodesic distance")

# -----------------------------------------------------------------------------
# Loss
loss_arg = add_argument_group("loss")
loss_arg.add_argument(
    "--weight_decay", type=float, default=0, help=""
    "l2 decay")
loss_arg.add_argument(
    "--momentum", type=float, default=0.9, help=""
    "momentum")
loss_arg.add_argument(
    "--loss_detector", type=float, default=1.0, help=""
    "weight of the detector loss")
loss_arg.add_argument(
    "--loss_classif", type=float, default=1.0, help=""
    "weight of the classification loss")
loss_arg.add_argument(
    "--loss_essential", type=float, default=0.1, help=""
    "weight of the essential loss")
loss_arg.add_argument(
    "--loss_essential_init_iter", type=int, default=20000, help=""
    "initial iterations to run only the classification loss")


# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument(
    "--run_mode", type=str, default="train", help=""
    "run_mode")
train_arg.add_argument(
    "--train_batch_size", type=int, default=4, help=""
    "batch size")
train_arg.add_argument(
    "--train_lr", type=float, default=1e-3, help=""
    "learning rate")
train_arg.add_argument(
    '--lr_step', type=int,  default=200000,
    help='Decrease step for learning rate.')
train_arg.add_argument(
    '--gamma', type=float, default=1, help='LR is multiplied by gamma on schedule.')
train_arg.add_argument(
    "--gpu_id", type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
train_arg.add_argument(
    "--num_processor", type=int, default=8, help='numbers of used cpu')
train_arg.add_argument(
    "--train_iter", type=int, default=500000, help=""
    "training iterations to perform")
train_arg.add_argument(
    "--log_base", type=str, default="./log/", help=""
    "save directory name inside results")
train_arg.add_argument(
    "--log_suffix", type=str, default="", help=""
    "suffix of log dir")
train_arg.add_argument(
    "--val_intv", type=int, default=10000, help=""
    "validation interval")
train_arg.add_argument(
    "--save_intv", type=int, default=1000, help=""
    "summary interval")
train_arg.add_argument(
    "--fix_backbone", type=str2bool, default=True, help=""
    "fix backbone network?")
train_arg.add_argument(
    "--fix_temp", type=str2bool, default=True, help=""
    "fix termperature ")
# -----------------------------------------------------------------------------
# Testing
test_arg = add_argument_group("Test")
test_arg.add_argument(
    "--use_ransac", type=str2bool, default=True, help=""
    "use ransac when testing?")
test_arg.add_argument(
    "--test_individually", type=str2bool, default=False, help=""
    "report results on each sequence")
test_arg.add_argument(
    "--model_path", type=str, default="", help=""
    "model path for test")
test_arg.add_argument(
    "--res_path", type=str, default="", help=""
    "path for saving results")
test_arg.add_argument(
    "--save_vis", type=str2bool, default=False, help=""
    "save visualization results")
test_arg.add_argument(
    "--vis_geod_th", type=float, default=1e-4, help=""
    "theshold for the good geodesic distance used in visualization")

# -----------------------------------------------------------------------------
# Visualization
vis_arg = add_argument_group('Visualization')
vis_arg.add_argument(
    "--tqdm_width", type=int, default=79, help=""
    "width of the tqdm bar"
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
