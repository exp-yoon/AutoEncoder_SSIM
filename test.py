import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import morphology 
from glob import glob
import cv2
import os

from utils import read_img, get_patch, patch2img, set_img_color, bg_mask
from network import AutoEncoder
from options import Options


cfg = Options().parse()

# network
autoencoder = AutoEncoder(cfg)

if cfg.weight_file:
    autoencoder.load_weights(cfg.chechpoint_dir + '/' + cfg.weight_file)
else:
    file_list = os.listdir(cfg.chechpoint_dir)
    latest_epoch = max([int(i.split('-')[0]) for i in file_list if 'hdf5' in i])
    print('load latest weight file: ', latest_epoch)
    autoencoder.load_weights(glob(cfg.chechpoint_dir + '/' + str(latest_epoch) + '*.hdf5')[0])
#autoencoder.summary()

def get_residual_map(img_path, cfg):
    test_img = read_img(img_path,cfg.grayscale) #이미지 gray scale로 불러오기

    #이미지 사이즈 맞춰주기
    if test_img.shape[:2] != (cfg.im_resize, cfg.im_resize):
        test_img = cv2.resize(test_img, (cfg.im_resize, cfg.im_resize))
    if cfg.im_resize != cfg.mask_size:
        tmp = (cfg.im_resize - cfg.mask_size)//2
        test_img = test_img[tmp:tmp+cfg.mask_size, tmp:tmp+cfg.mask_size]

    test_img_ = test_img/255.

    # autoencoder에서 결과 이미지 출력
    if test_img.shape[:2] == (cfg.patch_size, cfg.patch_size):
        test_img_ = np.expand_dims(test_img_, 0)
        decoded_img = autoencoder.predict(test_img_)
    else:
        patches = get_patch(test_img_, cfg.patch_size, cfg.stride)
        patches = autoencoder.predict(patches)
        decoded_img = patch2img(patches, cfg.im_resize, cfg.patch_size, cfg.stride)

    decoded_img = np.squeeze(decoded_img)
    decoded_img *= 255
    return decoded_img.astype(np.uint8)


def get_threshold(cfg):
    print('estimating threshold...')
    valid_good_list = glob(cfg.train_data_dir + '/*') # valid 이미지 불러오기
    num_valid_data = int(np.ceil(len(valid_good_list) * 0.2))
    total_rec_ssim, total_rec_l1 = [], []
    for img_path in valid_good_list[-num_valid_data:]:
        _, _, ssim_residual_map, l1_residual_map = get_residual_map(img_path, cfg)
        total_rec_ssim.append(ssim_residual_map)
        total_rec_l1.append(l1_residual_map)
    total_rec_ssim = np.array(total_rec_ssim)
    total_rec_l1 = np.array(total_rec_l1)
    # 백분위 분위수 구하는거..즉 이미지 내 pixel의 threshold/ cfg에 percent로 현재 98% 지정되어있음
    ssim_threshold = float(np.percentile(total_rec_ssim, [cfg.percent]))
    l1_threshold = float(np.percentile(total_rec_l1, [cfg.percent]))
    print('ssim_threshold: %f, l1_threshold: %f' %(ssim_threshold, l1_threshold))
    if not cfg.ssim_threshold:
        cfg.ssim_threshold = ssim_threshold
    if not cfg.l1_threshold:
        cfg.l1_threshold = l1_threshold


def get_depressing_mask(cfg):
    depr_mask = np.ones((cfg.mask_size, cfg.mask_size)) * 0.2
    depr_mask[5:cfg.mask_size-5, 5:cfg.mask_size-5] = 1
    cfg.depr_mask = depr_mask


def get_results(file_list, cfg):
    for img_path in file_list:
        img_name = img_path.split('\\')[-1][:-4]
        c = '' if not cfg.sub_folder else k
        decode_img = get_residual_map(img_path, cfg)

        decode_img = cv2.cvtColor(decode_img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(cfg.save_dir + '/' + img_name + '_prediction.bmp', decode_img)


if __name__ == '__main__':
    # if not cfg.ssim_threshold or not cfg.l1_threshold:
    #     get_threshold(cfg) # threshold 결정
    #
    # get_depressing_mask(cfg)

    if cfg.sub_folder:
        for k in cfg.sub_folder:
            test_list = glob(cfg.test_dir+'/'+k+'/*')
            get_results(test_list, cfg)
    else:
        test_list = glob(cfg.test_dir+'/*')
        get_results(test_list, cfg)
