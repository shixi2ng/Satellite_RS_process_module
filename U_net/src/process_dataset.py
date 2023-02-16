import os
from PIL import Image
import numpy as np
import gdal
from torch.utils.data import Dataset


class DriveDataset(Dataset):

    def __init__(self, root: str, train_test_factor: bool, transforms=None, in_channel=3, class_num=1):

        super(DriveDataset, self).__init__()

        self.flag = "train" if train_test_factor else "test"
        data_root = os.path.join(root, "Dataset", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms

        # Define the file list
        input_names = [i for i in os.listdir(os.path.join(data_root, "input")) if i.endswith(".tif") or i.endswith(".TIF")]
        self.input_list = [os.path.join(data_root, "input", i) for i in input_names]
        self.sample_list = [os.path.join(data_root, "sample", i.split(".")[0] + "_sample.TIF") for i in input_names]

        # check files
        for i in self.sample_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.in_channels = 1

    def __getitem__(self, index):

        input_ds = gdal.Open(self.img_list[index])
        input_ds_band = input_ds.RasterCount
        self.in_channels = input_ds_band
        input_array = np.zeros([input_ds.RasterYSize, input_ds.RasterXSize, input_ds_band])
        for i in range(1, input_ds_band + 1):
            input_array_temp = input_ds.GetRasterBand(i).ReadAsArray()
            input_array[:, :, i-1] = input_array_temp

        sample_ds = gdal.Open(self.img_list[index])
        sample_ds_band = sample_ds.RasterCount
        sample_array = np.zeros([sample_ds.RasterYSize, sample_ds.RasterXSize, sample_ds_band])
        for i in range(1, sample_ds_band + 1):
            sample_array_temp = sample_ds.GetRasterBand(i).ReadAsArray()
            sample_array[:, :, i - 1] = sample_array_temp

        mask_array = input_array[:, :, 0]
        mask_axis0 = np.nansum(mask_array, axis=0)
        mask_axis1 = np.nansum(mask_array, axis=1)

        x_min, x_max = np.min(np.argwhere(mask_axis0 != 0)), np.max(np.argwhere(mask_axis0 != 0))
        y_min, y_max = np.min(np.argwhere(mask_axis1 != 0)), np.max(np.argwhere(mask_axis1 != 0))

        input_array = input_array[y_min: y_max+1, x_min: x_max+1]
        sample_array = sample_array[y_min: y_max+1, x_min: x_max+1]
        mask_array = mask_array[y_min: y_max+1, x_min: x_max+1]
        sample_array[np.isnan(mask_array)] = 255

        # img = Image.open(self.img_list[index]).convert('RGB')
        # manual = Image.open(self.manual[index]).convert('L')
        #
        # manual = np.array(manual) / 255
        # roi_mask = Image.open(self.roi_mask[index]).convert('L')
        # roi_mask = 255 - np.array(roi_mask)
        # mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        sample_image = Image.fromarray(sample_array)
        input_array = Image.fromarray(input_array)

        if self.transforms is not None:
            input_img, sample_img = self.transforms(input_array, sample_image)

        return input_img, sample_img

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs