import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Detection")
    # 基本参数
    parser.add_argument("--cuda", action="store_true", default=False, help="use cuda.")
    parser.add_argument(
        "--tfboard", action="store_false", default=True, help="use tensorboard"
    )
    parser.add_argument(
        "--eval_epoch", type=int, default=10, help="interval between evaluations"
    )
    parser.add_argument(
        "--save_folder", default="weights/", type=str, help="Gamma update for SGD"
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of workers used in dataloading",
    )

    # 模型参数
    parser.add_argument("-v", "--version", default="yolo", help="yolo")

    # 训练配置
    parser.add_argument(
        "-bs", "--batch_size", default=8, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "-accu", "--accumulate", default=8, type=int, help="gradient accumulate."
    )
    parser.add_argument(
        "-no_wp",
        "--no_warm_up",
        action="store_true",
        default=False,
        help="yes or no to choose using warmup strategy to train",
    )
    parser.add_argument(
        "--wp_epoch", type=int, default=10, help="The upper bound of warm-up"
    )

    parser.add_argument(
        "--start_epoch", type=int, default=0, help="start epoch to train"
    )
    parser.add_argument("-r", "--resume", default=None, type=str, help="keep training")
    parser.add_argument(
        "-ms",
        "--multi_scale",
        action="store_true",
        default=False,
        help="use multi-scale trick",
    )
    parser.add_argument(
        "--max_epoch", type=int, default=150, help="The upper bound of warm-up"
    )
    parser.add_argument(
        "--lr_epoch", nargs="+", default=[90, 120], type=int, help="lr epoch to decay"
    )

    # 优化器参数
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="Momentum value for optim"
    )
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="Weight decay for SGD"
    )
    parser.add_argument("--gamma", default=0.1, type=float, help="Gamma update for SGD")

    # 数据集参数
    parser.add_argument("-d", "--dataset", default="coco", help="voc or coco")
    parser.add_argument("--root", default="data", help="data root")

    return parser.parse_args()
