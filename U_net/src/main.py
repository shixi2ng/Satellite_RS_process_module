import os
import time
import datetime
import torch
import basic_function as bf
import gdal
import numpy as np
from torch.utils.data import DataLoader

from unet import UNet
from train_utils.train_one_epoch import train_one_epoch, evaluate, create_lr_scheduler
from process_dataset import DriveDataset
import transform as T


class SegmentationPreset:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    if train:
        return SegmentationPreset(mean=mean, std=std)
    else:
        return SegmentationPreset(mean=mean, std=std)


def create_model(in_channels, num_classes, base_c):
    assert in_channels is int and num_classes is int and in_channels > 0 and num_classes > 0 and base_c is int and base_c > 0, f'Please input the in channel and classes as positive integer'
    model = UNet(in_channels, num_classes, base_channel=64)
    return model


class Unet4rs():
    def __init__(self, data_folder, rs_index_list, general_rs_index_folder, **kwargs):
        """

        :param data_folder: the rs data used to create the Unet model
        :param rs_index_list: the rs index used as the input
        :param general_rs_index_folder: rs index folder used to calculate the mean and std value for the study area
                                        whether the images were sampled or not.
        :param kwargs:
        """
        # Define argue
        self.in_channels = 1
        self.num_classes = 1
        self.device = 'cuda'
        self.batch_size = 4
        self.epochs = 200
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.print_freq = 1
        self.resume = ''
        self.start_epoch = 0
        self.save_best = True
        self.amp = False

        # Process kwargs
        self._process_kwargs(**kwargs)

        # Process args
        if type(rs_index_list) is list and type(general_rs_index_folder) is list and len(rs_index_list) == len(general_rs_index_folder):
            self.rs_index_list = rs_index_list
            self.general_rs_index_folder = general_rs_index_folder
        self.data_path = data_folder

    def _process_kwargs(self, **kwargs):
        for key_temp in kwargs.keys():
            if key_temp not in self.__dict__.keys():
                raise NameError(f'{str(key_temp)} is not defined argue!')
            elif type(kwargs[key_temp]) != self.__dict__[key_temp]:
                raise TypeError(f'{str(key_temp)} is not under right type!')
            else:
                self.__dict__[key_temp] = kwargs[key_temp]

    def _calculate_rs_mean_std_value(self, nodatavalue=np.nan):
        mean_list = []
        std_list = []
        for q in range(len(self.rs_index_list)):
            all_value = np.array([]).astype(np.float)
            rs_file_list = bf.file_filter(self.general_rs_index_folder[q], containing_word_list=[str(self.rs_index_list[q]), '.TIF'], and_or_factor='and', subfolder_detection=True)
            for file_temp in rs_file_list:
                file_ds = gdal.Open(file_temp)
                file_array = file_ds.GetRasterBand(1).ReadAsArray()
                file_array = file_array.flatten()
                file_array = np.delete(file_array, np.argwhere(np.isnan(file_array)))
                all_value = np.concatenate([all_value, file_array], axis=0)
            mean_list.append(np.nanmean(all_value))
            std_list.append(np.std(all_value))
        if len(mean_list) != len(self.rs_index_list) or len(std_list) != len(self.rs_index_list):
            raise Exception('Code error')
        return mean_list, std_list

    def run_module(self, **kwargs):
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        batch_size = self.batch_size
        # segmentation nun_classes + background
        num_classes = self.num_classes + 1

        # Compute the mean and std
        mean_list, std_list = self._calculate_rs_mean_std_value()

        # Store the output information
        bf.create_folder("results")
        results_file = "results\\results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Preprocess the train and valid dataset
        train_dataset = DriveDataset(self.data_path, train_test_factor=True, transforms=get_transform(train=True, mean=mean_list, std=std_list))
        val_dataset = DriveDataset(self.data_path, train_test_factor=False, transforms=get_transform(train=False, mean=mean_list, std=std_list))
        self.in_channels = train_dataset.in_channels

        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 collate_fn=val_dataset.collate_fn)

        model = create_model(self.in_channels, num_classes, base_c=64)
        model.to(device)

        params_to_optimize = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )

        scaler = torch.cuda.amp.GradScaler() if self.amp else None

        # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), self.epochs, warmup=True)

        if self.resume:
            checkpoint = torch.load(self.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            if self.amp:
                scaler.load_state_dict(checkpoint["scaler"])

        best_dice = 0.
        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                            lr_scheduler=lr_scheduler, print_freq=self.print_freq, scaler=scaler)

            confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
            val_info = str(confmat)
            print(val_info)
            print(f"dice coefficient: {dice:.3f}")
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n" \
                             f"dice coefficient: {dice:.3f}\n"
                f.write(train_info + val_info + "\n\n")

            if self.save_best is True:
                if best_dice < dice:
                    best_dice = dice
                else:
                    continue

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": self}
            if self.amp:
                save_file["scaler"] = scaler.state_dict()

            if self.save_best is True:
                torch.save(save_file, "save_weights/best_model.pth")
            else:
                torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))




def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = DriveDataset(args.data_path, train=True, transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    self = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(self)