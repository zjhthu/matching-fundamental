import argparse


arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

dataset_arg = add_argument_group("dataset")
dataset_arg.add_argument(
    "--dataset_path", type=str, default="../data/dataset/yfcc/")
dataset_arg.add_argument(
    "--dump_path", type=str, default="../data/")
dataset_arg.add_argument(
    "--dump_prefix", type=str, default="yfcc")
dataset_arg.add_argument(
    "--train_max_tr_sample", type=int, default=10000, help=""
    "number of max training samples")
dataset_arg.add_argument(
    "--train_max_va_sample", type=int, default=100, help=""
    "number of max validation samples")
dataset_arg.add_argument(
    "--train_max_te_sample", type=int, default=100, help=""
    "number of max test samples")
dataset_arg.add_argument(
    "--train_seqs", type=str, default='valid_train_seq.txt', help=""
    "train seq name")
dataset_arg.add_argument(
    "--test_max_te_sample", type=int, default=1000, help=""
    "number of max test samples in unseen scenes")
dataset_arg.add_argument(
    "--test_seqs", type=str, default="buckingham_palace.notre_dame_front_facade.sacre_coeur.reichstag")
dataset_arg.add_argument(
    "--vis_th", type=float, default=50, help=""
    "visibility threshold")

def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()
