from __future__ import division
import imageio
import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import config
import time
import copy
from rotation_3D import rotation3d, shear3d, rotation3d_itk
import json
import os
from utils.ran_brightness_contrast import RandomBrightnessContrast_corrected
from albumentations import Compose, RandomGamma
import cv2
import pandas as pd
import nibabel as nib
from enum import Enum


def json_load(path):
    """
    Load obj from json file
    """
    with open(path, 'r') as f:
        return json.load(f)
    return None


def load_img(filepath):
    img = imageio.imread(filepath)
    # img_shape = img.shape[1]
    # img_3d = img.reshape((-1, img_shape, img_shape))
    # img_3d = img_3d.transpose((1,2,0))
    return img


def load_img_path(directory):
    if directory.endswith('.png'):
        return load_img(directory)


def load_nii(load_fp):
    im = nib.load(str(load_fp))
    try:
        return np.asanyarray(im.dataobj), np.asanyarray(im.affine)
    except:
        return np.asanyarray(im.dataobj)


def load_nii_path(load_fp):
    img, _ = load_nii(load_fp)
    img = img[np.newaxis, :, :, :]
    return img


def do_crop(img, crop_size, crop_x, crop_y, crop_z):
    crop_xw = crop_x + crop_size[0]
    crop_yh = crop_y + crop_size[1]
    crop_zd = crop_z + crop_size[2]
    img = img[crop_z:crop_zd, crop_y: crop_yh, crop_x: crop_xw]
    return img


def constomized_RandomBrightnessContrast_aug(brightness_limit, contrast_limit, p=0.5):
    return Compose([
        RandomBrightnessContrast_corrected(brightness_limit=brightness_limit, contrast_limit=contrast_limit),
    ], p=p)


def constomized_RandomGamma_aug(gamma_limit, p=0.5):
    return Compose([
        RandomGamma(gamma_limit=gamma_limit),
    ], p=p)


class PixelWindow(Enum):
    NIL = (None, None)
    Lung = (-600, 1600)
    Bone = (400, 1600)
    Mediastinum = (40, 400)
    Aneurysm1 = (400, 1000)
    Artery1 = (400, 1200)
    LungMediastinum = (-580, 1640)


# 加窗宽窗位
def convert_window(image, window=PixelWindow.Lung, is_float=False, scale=255):
    window_center, window_width = window.value
    max_hu = window_center + window_width / 2
    min_hu = window_center - window_width / 2
    image_out = np.zeros(image.shape)
    image_out_16 = np.zeros(image.shape)
    w1 = (image > min_hu) & (image < max_hu)
    norm_to = float(scale)
    image_out[w1] = ((image[w1] - window_center + 0.5) / (window_width - 1.0) + 0.5) * norm_to
    image_out[image <= min_hu] = image[image <= min_hu] = 0.
    image_out[image >= max_hu] = image[image >= max_hu] = norm_to
    image_out_16[image <= min_hu] = min_hu
    image_out_16[image >= max_hu] = max_hu
    np_array = np.array(image_out)
    if is_float:
        return np_array / 255
    np_array = np_array.astype('uint8')
    return np_array


class DatasetFromList(data.Dataset):
    def __init__(self, pair_image_list, roi_list, opt):
        super(DatasetFromList, self).__init__()
        self.seg_filenames = []
        self.max_wh_list = []
        self.label_pos = opt['label_pos']
        self.final_size = opt['final_size']
        self.shear = opt['shear']
        self.rotation = opt['rotation']
        self.train_crop = opt['train_crop']
        self.random_crop = opt['random_crop']
        self.flip = opt['flip']
        self.offset_max = opt['offset_max']
        self.ran_zoom = opt['ran_zoom']
        self.train_flag = opt['train_flag']
        self.pad = opt['pad']
        self.normalize = opt['normalize']
        self.test_zoom = opt['test_zoom']
        self.use_mask = opt['use_mask']
        self.pre_crop = opt['pre_crop']
        self.black_out = opt['black_out']
        self.random_brightness_limit = opt['random_brightness_limit']
        self.random_contrast_limit = opt['random_contrast_limit']
        self.black_in = opt['black_in']
        self.new_black_out = opt['new_black_out']
        self.new_black_in = opt['new_black_in']
        self.TBMSL_NET_opt = opt['TBMSL_NET_opt']
        self.use_mask_oneslice = opt['use_mask_oneslice']
        self.clahe = opt['clahe']
        if self.clahe:
            self.clahe_apply = cv2.createCLAHE(clipLimit=self.clahe[0], tileGridSize=self.clahe[1])
        if type(self.use_mask_oneslice) == str and self.use_mask_oneslice.endswith('.json'):
            self.json_use_mask_oneslice = json_load(self.use_mask_oneslice)

        if self.TBMSL_NET_opt:
            self.TBMSL_NET_opt_json = json_load(self.TBMSL_NET_opt['json'])

        if self.black_out:
            self.black_out_list = []
            self.black_out_dict = json_load(self.black_out['json'])
            self.black_out_cor = []

        if self.black_in:
            self.black_in_list = []
            self.black_in_dict = json_load(self.black_out)
            self.black_in_cor = []
        if self.random_brightness_limit or self.random_contrast_limit:
            if not self.random_brightness_limit:
                self.random_brightness_limit = [0, 0]
            if not self.random_contrast_limit:
                self.random_contrast_limit = [0, 0]
            self.constomized_RandomBrightnessContrast = constomized_RandomBrightnessContrast_aug(
                brightness_limit=self.random_brightness_limit,
                contrast_limit=self.random_contrast_limit)
        self.random_gamma_limit = opt['random_gamma_limit']
        if self.random_gamma_limit:
            self.constomized_RandomGamma = constomized_RandomGamma_aug(gamma_limit=self.random_gamma_limit)

        data_info = pd.read_csv(pair_image_list)  # 数据地址处理
        self.image_filenames = data_info['mask_img'].tolist()
        self.label_list = data_info['label']

        ###dataaug
        self.center_crop = opt['center_crop']
        self.scale = opt['scale']
        print('len_list', len(self.image_filenames))

    def __getitem__(self, index):

        input = load_nii_path(self.image_filenames[index])
        label = self.label_list[index]

        input, label, mask, seg_mask = self.__data_aug(input, label, index)

        return torch.FloatTensor(input), label, self.image_filenames[index], index  # input_bbn, label_bbn, seg_mask

    def __len__(self):
        # print self.train_flag
        return len(self.image_filenames)

    def __data_aug(self, input, label, index):

        input_shape = input.shape[1]
        ############2D transform######################################
        if self.random_brightness_limit or self.random_contrast_limit:
            data = {"image": input, }
            data = self.constomized_RandomBrightnessContrast(**data)
            input = data['image']
        if self.random_gamma_limit:
            data = {"image": input, }
            data = self.constomized_RandomGamma(**data)
            input = data['image']
        input = input.reshape((-1, input_shape, input_shape))
        ori_input_shape = input.shape

        if self.black_out and label == 2:
            bk_cor = self.black_out_dict[self.image_filenames[index].replace(config.img_data_dir, '')]  # xyzwhd
            x1_bk, y1_bk, z1_bk, x2_bk, y2_bk, z2_bk = bk_cor[0], bk_cor[1], bk_cor[2], bk_cor[3] + bk_cor[0], bk_cor[
                4] + bk_cor[1], bk_cor[5] + bk_cor[2]
            input[z1_bk:z2_bk, y1_bk:y2_bk, x1_bk:x2_bk] = np.random.randint(0, 255)
            label = np.array([0.])

        if self.black_in and self.black_in_list[index] == 1:
            bk_cor = self.black_in_cor[index]  # xyzwhd
            x1_bk, y1_bk, z1_bk, x2_bk, y2_bk, z2_bk = bk_cor[0], bk_cor[1], bk_cor[2], bk_cor[3] + bk_cor[0], bk_cor[
                4] + bk_cor[1], bk_cor[5] + bk_cor[2]
            back_ground = np.zeros(input.shape)
            back_ground[:, :, :] = np.random.randint(0, 255)
            back_ground[z1_bk:z2_bk, y1_bk:y2_bk, x1_bk:x2_bk] = input[z1_bk:z2_bk, y1_bk:y2_bk, x1_bk:x2_bk]
            input = back_ground

        if self.new_black_out and label == 1 and np.random.uniform(low=0, high=1) < 0.5:
            mask_seg = load_img_path(self.new_black_out['seg'] + self.all_list[index].split(' ')[0])
            mask_seg_shape = mask_seg.shape[1]
            mask_seg = mask_seg.reshape((-1, mask_seg_shape, mask_seg_shape)).astype(np.float32)
            mask_seg[mask_seg > 1] = 1
            mask_doc_bbox = \
                np.load(self.new_black_out['doc_bbox'] + self.all_list[index].split(' ')[0].replace('.png', '.npz'))[
                    'data']
            mask_doc_bbox_shape = mask_doc_bbox.shape[1]
            mask_doc_bbox = mask_doc_bbox.reshape((-1, mask_doc_bbox_shape, mask_doc_bbox_shape)).astype(np.float32)
            mask_doc_bbox[mask_doc_bbox > 1] = 1
            mask_all = mask_seg * mask_doc_bbox
            if self.new_black_in and np.random.uniform(low=0, high=1) < 0.5:
                input = input * (mask_all) + (1 - mask_all) * np.random.randint(0, 255)
                label = 1
            else:
                input = input * (1 - mask_all) + mask_all * np.random.randint(0, 255)
                label = 0

        if self.pre_crop:
            pre_crop_x = np.shape(input)[2] // 2 - self.pre_crop[0] // 2
            pre_crop_y = np.shape(input)[1] // 2 - self.pre_crop[1] // 2
            pre_crop_z = np.shape(input)[0] // 2 - self.pre_crop[2] // 2
            input = do_crop(input, self.pre_crop, pre_crop_x, pre_crop_y, pre_crop_z)
            input = np.array(input)

        if self.ran_zoom and self.train_crop:
            ranzoom_x = np.random.uniform(low=self.ran_zoom[0], high=self.ran_zoom[1])
            ranzoom_y = np.random.uniform(low=self.ran_zoom[0], high=self.ran_zoom[1])
            ranzoom_z = np.random.uniform(low=self.ran_zoom[0], high=self.ran_zoom[1])
            zoom = [ranzoom_x, ranzoom_y, ranzoom_z]
        else:
            zoom = [1, 1, 1]

        if self.shear:
            hyx = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hzx = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hxy = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hzy = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hxz = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hyz = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            shear = [hyx, hzx, hxy, hzy, hxz, hyz]
        else:
            shear = [0, 0, 0, 0, 0, 0]
        if self.pad:
            input = np.array(input)
            input = np.pad(input, self.pad, 'edge')
        if self.rotation == 'big':
            R_x = (np.random.choice(2) * 2 - 1) * np.random.choice(360)
            R_y = (np.random.choice(2) * 2 - 1) * np.random.choice(360)
            R_z = (np.random.choice(2) * 2 - 1) * np.random.choice(360)
        else:
            R_x, R_y, R_z = 0, 0, 0
        if self.test_zoom:
            zoom = self.test_zoom

        if self.ran_zoom or self.rotation or self.shear or self.test_zoom:
            input = rotation3d_itk(input, R_x, R_y, R_z, zoom, shear)
            input = np.array(input)

        if self.train_crop:
            if self.random_crop:
                shift_x = (np.random.choice(2) * 2 - 1) * np.random.choice(self.random_crop[0] + 1)
                shift_y = (np.random.choice(2) * 2 - 1) * np.random.choice(self.random_crop[1] + 1)
                shift_z = (np.random.choice(2) * 2 - 1) * np.random.choice(self.random_crop[2] + 1)
            else:
                shift_x, shift_y, shift_z = 0, 0, 0

            crop_x = np.shape(input)[2] // 2 - self.train_crop[0] // 2 + shift_x  # + offset_w//2
            crop_y = np.shape(input)[1] // 2 - self.train_crop[1] // 2 + shift_y  # + offset_h//2
            crop_z = np.shape(input)[0] // 2 - self.train_crop[2] // 2 + shift_z  # + offset_d//2

            input = do_crop(input, self.train_crop, crop_x, crop_y, crop_z)
            input = np.array(input)

        if self.center_crop:
            crop_x = np.shape(input)[2] // 2 - self.center_crop[0] // 2
            crop_y = np.shape(input)[1] // 2 - self.center_crop[1] // 2
            crop_z = np.shape(input)[0] // 2 - self.center_crop[2] // 2
            input = do_crop(input, self.center_crop, crop_x, crop_y, crop_z)
            input = np.array(input)

        if self.flip:
            flip_x = np.random.choice(2) * 2 - 1
            flip_y = np.random.choice(2) * 2 - 1
            flip_z = np.random.choice(2) * 2 - 1
            input = input[::flip_z, ::flip_y, ::flip_x]

        mask = []
        if self.use_mask:
            mask = load_img_path(self.image_filenames[index].replace('.png', '') + '_mask.png')
            mask_shape = mask.shape[1]
            mask = mask.reshape((-1, mask_shape, mask_shape)).astype(np.float32)

            if self.pre_crop:
                mask = do_crop(mask, self.pre_crop, pre_crop_x, pre_crop_y, pre_crop_z)
            if self.train_crop:
                mask = do_crop(mask, self.train_crop, crop_x, crop_y, crop_z)
            elif self.center_crop:
                mask = do_crop(mask, self.center_crop, crop_x, crop_y, crop_z)
            if type(self.flip) == list:
                mask = mask[::flip_z, ::flip_y, ::flip_x]
            mask[mask > 0] = 1

            # slice_num = mask.shape[0]
            # middle = slice_num // 2
            # mask[:middle - 4, :, :] = 0
            # mask[middle + +1 + 4:, :, :] = 0

            mask = torch.from_numpy(mask.copy())

        mask_oneslice = []
        if self.use_mask_oneslice:

            if type(self.use_mask_oneslice) != str:
                mask_oneslice = load_img_path(self.image_filenames[index].replace('.png', '') + '_mask.png')
                mask_oneslice_shape = mask_oneslice.shape[1]
                mask_oneslice = mask_oneslice.reshape((-1, mask_oneslice_shape, mask_oneslice_shape)).astype(np.float32)
                mask_oneslice[mask_oneslice > 0] = 1
                slice_num = mask_oneslice.shape[0]
                middle = slice_num // 2

                mask_oneslice[:middle, :, :] = 0
                mask_oneslice[middle + 1:, :, :] = 0
                # mask_oneslice[middle + 1, :, :] = 0

            elif type(self.use_mask_oneslice) == str and self.use_mask_oneslice.endswith('.json'):

                one_mask_oneslice_cube = self.json_use_mask_oneslice[
                    self.image_filenames[index].replace(config.img_data_dir, '')]  # xyzwhd
                mask_oneslice = np.zeros(ori_input_shape, dtype=np.float32)
                mask_oneslice[
                one_mask_oneslice_cube[2]:one_mask_oneslice_cube[2] + one_mask_oneslice_cube[5],
                one_mask_oneslice_cube[1]:one_mask_oneslice_cube[1] + one_mask_oneslice_cube[4],
                one_mask_oneslice_cube[0]:one_mask_oneslice_cube[0] + one_mask_oneslice_cube[3],
                ] = 1
            if self.pre_crop:
                mask_oneslice = do_crop(mask_oneslice, self.pre_crop, pre_crop_x, pre_crop_y, pre_crop_z)
            if self.train_crop:
                mask_oneslice = do_crop(mask_oneslice, self.train_crop, crop_x, crop_y, crop_z)
            elif self.center_crop:
                mask_oneslice = do_crop(mask_oneslice, self.center_crop, crop_x, crop_y, crop_z)
            if type(self.flip) == list:
                mask_oneslice = mask_oneslice[::flip_z, ::flip_y, ::flip_x]
            mask_oneslice = torch.from_numpy(mask_oneslice.copy())
        if self.clahe:
            input = input.reshape((-1, self.final_size[0]))  # self.final_size x y z
            input = self.clahe_apply.apply(input)
            input = input.reshape(self.final_size[::-1])

        input = input * self.scale
        input = torch.from_numpy(input)
        if self.normalize:
            input = (input - 0.5) / 0.5

        input = input.unsqueeze(0)
        if self.use_mask_oneslice:
            input = torch.cat((input, input * mask_oneslice, input * (1 - mask_oneslice)), dim=0)
        if self.use_mask:
            input = torch.cat((input, input * mask, input * (1 - mask)), dim=0)

        if 'bt_' in config.data_mode:
            input = input.transpose((2, 1, 0))

        seg_mask = []
        if self.TBMSL_NET_opt and self.TBMSL_NET_opt['seg_mask_4input'] == 'original':
            mask = load_img_path(self.TBMSL_NET_opt['seg'] + self.all_list[index].split(' ')[0])
            mask_shape = mask.shape[1]
            mask = mask.reshape((-1, mask_shape, mask_shape)).astype(np.float32)
            if self.train_crop:
                mask = do_crop(mask, self.train_crop, crop_x, crop_y, crop_z)
            elif self.center_crop:
                mask = do_crop(mask, self.center_crop, crop_x, crop_y, crop_z)
            if self.flip:
                mask = mask[::flip_z, ::flip_y, ::flip_x]
            mask[mask > 0] = 1
            mask_all = mask

            if label == 1:
                cor_gt_3d = self.TBMSL_NET_opt_json[self.all_list[index].split(' ')[0]]
                x1_bk, y1_bk, z1_bk, x2_bk, y2_bk, z2_bk = cor_gt_3d[0], cor_gt_3d[1], cor_gt_3d[2], cor_gt_3d[3] + \
                                                           cor_gt_3d[0], \
                                                           cor_gt_3d[4] + cor_gt_3d[1], cor_gt_3d[5] + cor_gt_3d[2]
                mask_doc_bbox = np.zeros((9, 112, 112))
                mask_doc_bbox[z1_bk:z2_bk, y1_bk:y2_bk, x1_bk:x2_bk] = 1

                if self.train_crop:
                    mask_doc_bbox = do_crop(mask_doc_bbox, self.train_crop, crop_x, crop_y, crop_z)
                elif self.center_crop:
                    mask_doc_bbox = do_crop(mask_doc_bbox, self.center_crop, crop_x, crop_y, crop_z)
                if self.flip:
                    mask_doc_bbox = mask_doc_bbox[::flip_z, ::flip_y, ::flip_x]
                mask_all = mask_doc_bbox * mask_all

            seg_mask = torch.from_numpy(mask_all.copy().astype(np.float32)).unsqueeze(0)
        return input, label, mask, seg_mask,

    def _get_random_params(self, offset_max, index=0):
        # np.random.seed(int(time.time() + 1e5 * index))
        rand_x = np.random.rand()
        rand_y = np.random.rand()
        rand_z = np.random.rand()
        rand_w = np.random.rand()
        rand_h = np.random.rand()
        rand_r = np.random.rand()
        rand_lr = np.random.rand()
        rand_td = np.random.rand()
        rand_r1 = np.random.rand()

        offset_x = int((rand_x * 2 - 1) * offset_max * 0.3)
        offset_y = int((rand_y * 2 - 1) * offset_max * 0.3)
        offset_z = int((rand_z * 2 - 1) * offset_max * 0.3)
        offset_w = int(((rand_w + 0.25) * 2 - 1) * offset_max * 2)
        offset_h = int(((rand_h + 0.25) * 2 - 1) * offset_max * 2)  # -10 30
        offset_d = int(((rand_h + 0.25) * 2 - 1) * offset_max * 2)
        # offset_w = int(((rand_w) * 2 - 1) * offset_max * 2)
        # offset_h = int(((rand_h) * 2 - 1) * offset_max * 2)
        rand_angle = int(rand_r * 360)

        return offset_x, offset_y, offset_z, offset_w, offset_h, offset_d, rand_angle, rand_lr, rand_td, rand_r1


class DatasetFromList_rad(data.Dataset):
    def __init__(self, pair_image_list, rad_list, opt):
        super(DatasetFromList, self).__init__()
        self.seg_filenames = []
        self.max_wh_list = []
        self.label_pos = opt['label_pos']
        self.final_size = opt['final_size']
        self.shear = opt['shear']
        self.rotation = opt['rotation']
        self.train_crop = opt['train_crop']
        self.random_crop = opt['random_crop']
        self.flip = opt['flip']
        self.offset_max = opt['offset_max']
        self.ran_zoom = opt['ran_zoom']
        self.train_flag = opt['train_flag']
        self.pad = opt['pad']
        self.normalize = opt['normalize']
        self.test_zoom = opt['test_zoom']
        self.use_mask = opt['use_mask']
        self.pre_crop = opt['pre_crop']
        self.black_out = opt['black_out']
        self.random_brightness_limit = opt['random_brightness_limit']
        self.random_contrast_limit = opt['random_contrast_limit']
        self.black_in = opt['black_in']
        self.new_black_out = opt['new_black_out']
        self.new_black_in = opt['new_black_in']
        self.TBMSL_NET_opt = opt['TBMSL_NET_opt']
        self.use_mask_oneslice = opt['use_mask_oneslice']
        self.clahe = opt['clahe']
        if self.clahe:
            self.clahe_apply = cv2.createCLAHE(clipLimit=self.clahe[0], tileGridSize=self.clahe[1])
        if type(self.use_mask_oneslice) == str and self.use_mask_oneslice.endswith('.json'):
            self.json_use_mask_oneslice = json_load(self.use_mask_oneslice)

        if self.TBMSL_NET_opt:
            self.TBMSL_NET_opt_json = json_load(self.TBMSL_NET_opt['json'])

        if self.black_out:
            self.black_out_list = []
            self.black_out_dict = json_load(self.black_out['json'])
            self.black_out_cor = []

        if self.black_in:
            self.black_in_list = []
            self.black_in_dict = json_load(self.black_out)
            self.black_in_cor = []
        if self.random_brightness_limit or self.random_contrast_limit:
            if not self.random_brightness_limit:
                self.random_brightness_limit = [0, 0]
            if not self.random_contrast_limit:
                self.random_contrast_limit = [0, 0]
            self.constomized_RandomBrightnessContrast = constomized_RandomBrightnessContrast_aug(
                brightness_limit=self.random_brightness_limit,
                contrast_limit=self.random_contrast_limit)
        self.random_gamma_limit = opt['random_gamma_limit']
        if self.random_gamma_limit:
            self.constomized_RandomGamma = constomized_RandomGamma_aug(gamma_limit=self.random_gamma_limit)

        data_info = pd.read_csv(pair_image_list)  #
        self.image_filenames = data_info['mask_img'].tolist()
        self.label_list = data_info['label']
        rad_info = pd.read_csv(rad_list)  #
        self.rad_filenames = rad_info['mask_img'].tolist()
        self.rad_feature = np.array(rad_info)[:, 1:]

        ###dataaug
        self.center_crop = opt['center_crop']
        self.scale = opt['scale']
        print('len_list', len(self.image_filenames))

    def __getitem__(self, index):

        input = load_nii_path(self.image_filenames[index])
        rad = self.rad_feature[self.rad_filenames.index(self.image_filenames[index])]
        label = self.label_list[index]

        input, label, mask, seg_mask = self.__data_aug(input, label, index)

        return torch.FloatTensor(input), torch.FloatTensor(rad), label, self.image_filenames[index], index  

    def __len__(self):
        # print self.train_flag
        return len(self.image_filenames)

    def __data_aug(self, input, label, index):

        input_shape = input.shape[1]
        ############2D transform######################################
        if self.random_brightness_limit or self.random_contrast_limit:
            data = {"image": input, }
            data = self.constomized_RandomBrightnessContrast(**data)
            input = data['image']
        if self.random_gamma_limit:
            data = {"image": input, }
            data = self.constomized_RandomGamma(**data)
            input = data['image']
        input = input.reshape((-1, input_shape, input_shape))
        ori_input_shape = input.shape

        if self.black_out and label == 2:
            bk_cor = self.black_out_dict[self.image_filenames[index].replace(config.img_data_dir, '')]  # xyzwhd
            x1_bk, y1_bk, z1_bk, x2_bk, y2_bk, z2_bk = bk_cor[0], bk_cor[1], bk_cor[2], bk_cor[3] + bk_cor[0], bk_cor[
                4] + bk_cor[1], bk_cor[5] + bk_cor[2]
            input[z1_bk:z2_bk, y1_bk:y2_bk, x1_bk:x2_bk] = np.random.randint(0, 255)
            label = np.array([0.])

        if self.black_in and self.black_in_list[index] == 1:
            bk_cor = self.black_in_cor[index]  # xyzwhd
            x1_bk, y1_bk, z1_bk, x2_bk, y2_bk, z2_bk = bk_cor[0], bk_cor[1], bk_cor[2], bk_cor[3] + bk_cor[0], bk_cor[
                4] + bk_cor[1], bk_cor[5] + bk_cor[2]
            back_ground = np.zeros(input.shape)
            back_ground[:, :, :] = np.random.randint(0, 255)
            back_ground[z1_bk:z2_bk, y1_bk:y2_bk, x1_bk:x2_bk] = input[z1_bk:z2_bk, y1_bk:y2_bk, x1_bk:x2_bk]
            input = back_ground

        if self.new_black_out and label == 1 and np.random.uniform(low=0, high=1) < 0.5:
            mask_seg = load_img_path(self.new_black_out['seg'] + self.all_list[index].split(' ')[0])
            mask_seg_shape = mask_seg.shape[1]
            mask_seg = mask_seg.reshape((-1, mask_seg_shape, mask_seg_shape)).astype(np.float32)
            mask_seg[mask_seg > 1] = 1
            mask_doc_bbox = \
                np.load(self.new_black_out['doc_bbox'] + self.all_list[index].split(' ')[0].replace('.png', '.npz'))[
                    'data']
            mask_doc_bbox_shape = mask_doc_bbox.shape[1]
            mask_doc_bbox = mask_doc_bbox.reshape((-1, mask_doc_bbox_shape, mask_doc_bbox_shape)).astype(np.float32)
            mask_doc_bbox[mask_doc_bbox > 1] = 1
            mask_all = mask_seg * mask_doc_bbox
            if self.new_black_in and np.random.uniform(low=0, high=1) < 0.5:
                input = input * (mask_all) + (1 - mask_all) * np.random.randint(0, 255)
                label = 1
            else:
                input = input * (1 - mask_all) + mask_all * np.random.randint(0, 255)
                label = 0

        if self.pre_crop:
            pre_crop_x = np.shape(input)[2] // 2 - self.pre_crop[0] // 2
            pre_crop_y = np.shape(input)[1] // 2 - self.pre_crop[1] // 2
            pre_crop_z = np.shape(input)[0] // 2 - self.pre_crop[2] // 2
            input = do_crop(input, self.pre_crop, pre_crop_x, pre_crop_y, pre_crop_z)
            input = np.array(input)

        if self.ran_zoom and self.train_crop:
            ranzoom_x = np.random.uniform(low=self.ran_zoom[0], high=self.ran_zoom[1])
            ranzoom_y = np.random.uniform(low=self.ran_zoom[0], high=self.ran_zoom[1])
            ranzoom_z = np.random.uniform(low=self.ran_zoom[0], high=self.ran_zoom[1])
            zoom = [ranzoom_x, ranzoom_y, ranzoom_z]
        else:
            zoom = [1, 1, 1]

        if self.shear:
            hyx = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hzx = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hxy = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hzy = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hxz = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hyz = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            shear = [hyx, hzx, hxy, hzy, hxz, hyz]
        else:
            shear = [0, 0, 0, 0, 0, 0]
        if self.pad:
            input = np.array(input)
            input = np.pad(input, self.pad, 'edge')
        if self.rotation == 'big':
            R_x = (np.random.choice(2) * 2 - 1) * np.random.choice(360)
            R_y = (np.random.choice(2) * 2 - 1) * np.random.choice(360)
            R_z = (np.random.choice(2) * 2 - 1) * np.random.choice(360)
        else:
            R_x, R_y, R_z = 0, 0, 0
        if self.test_zoom:
            zoom = self.test_zoom

        if self.ran_zoom or self.rotation or self.shear or self.test_zoom:
            input = rotation3d_itk(input, R_x, R_y, R_z, zoom, shear)
            input = np.array(input)

        if self.train_crop:
            if self.random_crop:
                shift_x = (np.random.choice(2) * 2 - 1) * np.random.choice(self.random_crop[0] + 1)
                shift_y = (np.random.choice(2) * 2 - 1) * np.random.choice(self.random_crop[1] + 1)
                shift_z = (np.random.choice(2) * 2 - 1) * np.random.choice(self.random_crop[2] + 1)
            else:
                shift_x, shift_y, shift_z = 0, 0, 0

            crop_x = np.shape(input)[2] // 2 - self.train_crop[0] // 2 + shift_x  # + offset_w//2
            crop_y = np.shape(input)[1] // 2 - self.train_crop[1] // 2 + shift_y  # + offset_h//2
            crop_z = np.shape(input)[0] // 2 - self.train_crop[2] // 2 + shift_z  # + offset_d//2

            input = do_crop(input, self.train_crop, crop_x, crop_y, crop_z)
            input = np.array(input)

        if self.center_crop:
            crop_x = np.shape(input)[2] // 2 - self.center_crop[0] // 2
            crop_y = np.shape(input)[1] // 2 - self.center_crop[1] // 2
            crop_z = np.shape(input)[0] // 2 - self.center_crop[2] // 2
            input = do_crop(input, self.center_crop, crop_x, crop_y, crop_z)
            input = np.array(input)

        if self.flip:
            flip_x = np.random.choice(2) * 2 - 1
            flip_y = np.random.choice(2) * 2 - 1
            flip_z = np.random.choice(2) * 2 - 1
            input = input[::flip_z, ::flip_y, ::flip_x]

        mask = []
        if self.use_mask:
            mask = load_img_path(self.image_filenames[index].replace('.png', '') + '_mask.png')
            mask_shape = mask.shape[1]
            mask = mask.reshape((-1, mask_shape, mask_shape)).astype(np.float32)

            if self.pre_crop:
                mask = do_crop(mask, self.pre_crop, pre_crop_x, pre_crop_y, pre_crop_z)
            if self.train_crop:
                mask = do_crop(mask, self.train_crop, crop_x, crop_y, crop_z)
            elif self.center_crop:
                mask = do_crop(mask, self.center_crop, crop_x, crop_y, crop_z)
            if type(self.flip) == list:
                mask = mask[::flip_z, ::flip_y, ::flip_x]
            mask[mask > 0] = 1

            # slice_num = mask.shape[0]
            # middle = slice_num // 2
            # mask[:middle - 4, :, :] = 0
            # mask[middle + +1 + 4:, :, :] = 0

            mask = torch.from_numpy(mask.copy())

        mask_oneslice = []
        if self.use_mask_oneslice:

            if type(self.use_mask_oneslice) != str:
                mask_oneslice = load_img_path(self.image_filenames[index].replace('.png', '') + '_mask.png')
                mask_oneslice_shape = mask_oneslice.shape[1]
                mask_oneslice = mask_oneslice.reshape((-1, mask_oneslice_shape, mask_oneslice_shape)).astype(np.float32)
                mask_oneslice[mask_oneslice > 0] = 1
                slice_num = mask_oneslice.shape[0]
                middle = slice_num // 2

                mask_oneslice[:middle, :, :] = 0
                mask_oneslice[middle + 1:, :, :] = 0
                # mask_oneslice[middle + 1, :, :] = 0

            elif type(self.use_mask_oneslice) == str and self.use_mask_oneslice.endswith('.json'):

                one_mask_oneslice_cube = self.json_use_mask_oneslice[
                    self.image_filenames[index].replace(config.img_data_dir, '')]  # xyzwhd
                mask_oneslice = np.zeros(ori_input_shape, dtype=np.float32)
                mask_oneslice[
                one_mask_oneslice_cube[2]:one_mask_oneslice_cube[2] + one_mask_oneslice_cube[5],
                one_mask_oneslice_cube[1]:one_mask_oneslice_cube[1] + one_mask_oneslice_cube[4],
                one_mask_oneslice_cube[0]:one_mask_oneslice_cube[0] + one_mask_oneslice_cube[3],
                ] = 1
            if self.pre_crop:
                mask_oneslice = do_crop(mask_oneslice, self.pre_crop, pre_crop_x, pre_crop_y, pre_crop_z)
            if self.train_crop:
                mask_oneslice = do_crop(mask_oneslice, self.train_crop, crop_x, crop_y, crop_z)
            elif self.center_crop:
                mask_oneslice = do_crop(mask_oneslice, self.center_crop, crop_x, crop_y, crop_z)
            if type(self.flip) == list:
                mask_oneslice = mask_oneslice[::flip_z, ::flip_y, ::flip_x]
            mask_oneslice = torch.from_numpy(mask_oneslice.copy())
        if self.clahe:
            input = input.reshape((-1, self.final_size[0]))  # self.final_size x y z
            input = self.clahe_apply.apply(input)
            input = input.reshape(self.final_size[::-1])

        input = input * self.scale
        input = torch.from_numpy(input)
        if self.normalize:
            input = (input - 0.5) / 0.5

        input = input.unsqueeze(0)
        if self.use_mask_oneslice:
            input = torch.cat((input, input * mask_oneslice, input * (1 - mask_oneslice)), dim=0)
        if self.use_mask:
            input = torch.cat((input, input * mask, input * (1 - mask)), dim=0)

        if 'bt_' in config.data_mode:
            input = input.transpose((2, 1, 0))

        seg_mask = []
        if self.TBMSL_NET_opt and self.TBMSL_NET_opt['seg_mask_4input'] == 'original':
            mask = load_img_path(self.TBMSL_NET_opt['seg'] + self.all_list[index].split(' ')[0])
            mask_shape = mask.shape[1]
            mask = mask.reshape((-1, mask_shape, mask_shape)).astype(np.float32)
            if self.train_crop:
                mask = do_crop(mask, self.train_crop, crop_x, crop_y, crop_z)
            elif self.center_crop:
                mask = do_crop(mask, self.center_crop, crop_x, crop_y, crop_z)
            if self.flip:
                mask = mask[::flip_z, ::flip_y, ::flip_x]
            mask[mask > 0] = 1
            mask_all = mask

            if label == 1:
                cor_gt_3d = self.TBMSL_NET_opt_json[self.all_list[index].split(' ')[0]]
                x1_bk, y1_bk, z1_bk, x2_bk, y2_bk, z2_bk = cor_gt_3d[0], cor_gt_3d[1], cor_gt_3d[2], cor_gt_3d[3] + \
                                                           cor_gt_3d[0], \
                                                           cor_gt_3d[4] + cor_gt_3d[1], cor_gt_3d[5] + cor_gt_3d[2]
                mask_doc_bbox = np.zeros((9, 112, 112))
                mask_doc_bbox[z1_bk:z2_bk, y1_bk:y2_bk, x1_bk:x2_bk] = 1

                if self.train_crop:
                    mask_doc_bbox = do_crop(mask_doc_bbox, self.train_crop, crop_x, crop_y, crop_z)
                elif self.center_crop:
                    mask_doc_bbox = do_crop(mask_doc_bbox, self.center_crop, crop_x, crop_y, crop_z)
                if self.flip:
                    mask_doc_bbox = mask_doc_bbox[::flip_z, ::flip_y, ::flip_x]
                mask_all = mask_doc_bbox * mask_all

            seg_mask = torch.from_numpy(mask_all.copy().astype(np.float32)).unsqueeze(0)
        return input, label, mask, seg_mask,

    def _get_random_params(self, offset_max, index=0):
        # np.random.seed(int(time.time() + 1e5 * index))
        rand_x = np.random.rand()
        rand_y = np.random.rand()
        rand_z = np.random.rand()
        rand_w = np.random.rand()
        rand_h = np.random.rand()
        rand_r = np.random.rand()
        rand_lr = np.random.rand()
        rand_td = np.random.rand()
        rand_r1 = np.random.rand()

        offset_x = int((rand_x * 2 - 1) * offset_max * 0.3)
        offset_y = int((rand_y * 2 - 1) * offset_max * 0.3)
        offset_z = int((rand_z * 2 - 1) * offset_max * 0.3)
        offset_w = int(((rand_w + 0.25) * 2 - 1) * offset_max * 2)
        offset_h = int(((rand_h + 0.25) * 2 - 1) * offset_max * 2)  # -10 30
        offset_d = int(((rand_h + 0.25) * 2 - 1) * offset_max * 2)
        # offset_w = int(((rand_w) * 2 - 1) * offset_max * 2)
        # offset_h = int(((rand_h) * 2 - 1) * offset_max * 2)
        rand_angle = int(rand_r * 360)

        return offset_x, offset_y, offset_z, offset_w, offset_h, offset_d, rand_angle, rand_lr, rand_td, rand_r1