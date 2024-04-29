from utils.parse import parse_args
import torch
import torch.nn as nn
from datasets.dataset_build import build_dataset, gt_create
from utils.misc import detection_collate
from models.yolov1 import Yolo_V1
import time
import os
import torch.optim as optim


def train():
    args = parse_args()
    print("----------------------------------------------------------")

    if args.cuda:
        print("Use cuda!")
        device = torch.device("cuda")
    else:
        print("Use cpu!")
        device = torch.device("cpu")
    if args.tfboard:
        print("use tensorboard")
        from torch.utils.tensorboard import SummaryWriter

        c_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        log_path = os.path.join("log/coco/", args.version, c_time)
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_path)
    if args.resume is not None:
        print(f"load checkpoint from{args.resume}!")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    train_size, val_size = 416, 416
    dataset, num_classes, evaluator = build_dataset(args, device, train_size, val_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=detection_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Yolo_V1(
        device=device,
        img_size=train_size,
        num_classes=num_classes,
        conf_th=0.1,
        nms_th=0.5,
        train=True,
    )
    model = model.to(device=device).train()
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    max_epoch = args.max_epoch
    lr_epoch = args.lr_epoch
    epoch_size = len(dataloader)
    best_map = -1.0
    t0 = time.time()
    for epoch in range(args.start_epoch, max_epoch):
        if epoch in lr_epoch:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
        for iter_i, (images, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    nw = args.wp_epoch * epoch_size
                    tmp_lr = base_lr * pow(ni * 1.0 / nw, 4)
                    set_lr(optimizer, tmp_lr)
                elif epoch == args.wp_epoch:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
            targets = [label.tolist() for label in targets]
            targets = gt_create(train_size, model.stride, targets)
            images = images.to(device)
            targets = targets.to(device)
            conf_loss, cls_loss, bbox_loss, total_loss = model(images, targets=targets)
            total_loss /= args.accumulate
            total_loss.backward()

            if ni % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
            if iter_i % 100 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar(
                        "obj loss", conf_loss.item(), iter_i + epoch * epoch_size
                    )
                    writer.add_scalar(
                        "cls loss", cls_loss.item(), iter_i + epoch * epoch_size
                    )
                    writer.add_scalar(
                        "box loss", bbox_loss.item(), iter_i + epoch * epoch_size
                    )

                t1 = time.time()
                print(
                    "[Epoch %d/%d][Iter %d/%d][lr %.6f]"
                    "[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]"
                    % (
                        epoch + 1,
                        max_epoch,
                        iter_i,
                        epoch_size,
                        tmp_lr,
                        conf_loss.item(),
                        cls_loss.item(),
                        bbox_loss.item(),
                        total_loss.item(),
                        train_size,
                        t1 - t0,
                    ),
                    flush=True,
                )

                t0 = time.time()
            # evaluation
        if epoch % args.eval_epoch == 0:
            model.is_train = False
            # model.set_grid(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            # convert to training mode.
            model.is_train = True
            # model.set_grid(train_size)
            model.train()

            cur_map = evaluator.map
            if args.tfboard:
                # viz loss
                writer.add_scalar("obj cur_map", cur_map, epoch)

            if cur_map > best_map:
                # update best-map
                best_map = cur_map
                # save model
                print("Saving state, epoch:", epoch + 1)
                weight_name = "{}_epoch_{}_{:.1f}.pth".format(
                    args.version, epoch + 1, best_map * 100
                )
                checkpoint_path = os.path.join(args.save_folder, weight_name)
                torch.save(model.state_dict(), checkpoint_path)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    train()
