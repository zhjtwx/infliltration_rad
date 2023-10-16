# -*- coding: UTF-8 -*-
import argparse
import pandas as pd
import csv
import logging
import os
import re
import shutil
import tempfile
import threading
import signal
import numpy as np
import SimpleITK as sitk
from datetime import datetime
from collections import OrderedDict
from multiprocessing import cpu_count, Pool
import tools
from tools import get_compact_range, info_filter
import radiomics
import glob
TEMP_DIR = tempfile.mkdtemp()
import cv2


# 调整CT图像的窗宽窗位
def window(img):
    win_min = -600
    win_max = 1500

    for i in range(img.shape[0]):
        img[i] = 255.0 * (img[i] - win_min) / (win_max - win_min)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255
        img[i] = img[i] - img[i].min()
        c = float(255) / img[i].max()
        img[i] = img[i] * c

    return img.astype(np.uint8)


def extract_feature(case):
    feature_vector = OrderedDict(case)

    try:
        img_dir = case['image']
        mask_nii = case['mask']
        use_dicom = case['use_dicom']
        img_reader = case['img_reader']
        use_pyradiomics = case['use_pyradiomics']
        if use_pyradiomics:
            from radiomics.featureextractor import RadiomicsFeatureExtractor
        else:
            from radiomics_ria.featureextractor import RadiomicsFeaturesExtractor
        threading.current_thread().name = mask_nii

        case_id = mask_nii.replace('/', '_')

        filename = r'features_' + str(case_id) + '.csv'
        output_filename = os.path.join(TEMP_DIR, filename)

        t = datetime.now()

        # mask
        single_reader = sitk.ImageFileReader()
        single_reader.SetFileName(mask_nii)
        mask = single_reader.Execute()
        mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
        mask_arr = np.where(mask_arr > 1, 1, mask_arr)
        # 如果mask有很少的voxel，则略过
        voxels = np.sum(mask_arr)
        if voxels <= 10:
            delta_t = datetime.now() - t
            logging.getLogger('radiomics_s.batch').error('Case: %s %s %s processed in %s PID %s (%s)',
                                                       case_id, case["image"], case["mask"], delta_t, os.getpid(),
                                                       "Mask only contains few segmented voxel! Ignored.")
            return feature_vector

        # 读取image
        if use_dicom:
            dicom_reader = sitk.ImageSeriesReader()
            dicom_reader.SetFileNames(dicom_reader.GetGDCMSeriesFileNames(img_dir))

            # fnames = [os.path.join(img_dir, x) for x in sorted(os.listdir(img_dir), key=lambda r: int(list(filter(str.isdigit, re.split('\.|_|-', r)))[-1]))]
            # dicom_reader.SetFileNames(fnames)
            try:
                img = dicom_reader.Execute()
            except:
                fnames = glob.glob(img_dir + '/*.dcm')

                dicom_reader.SetFileNames(fnames)

        else:
            single_reader.SetFileName(img_dir)
            if "nrrd" in img_reader:
                single_reader.SetImageIO("NrrdImageIO")
            else:
                single_reader.SetImageIO("NiftiImageIO")
            img = single_reader.Execute()
        img_arr = sitk.GetArrayFromImage(img)
        # img_arr = window(img_arr)
        img_arr = np.transpose(img_arr, (2, 1, 0))
        mask_arr = np.transpose(mask_arr, (2, 1, 0))

        # print(img_arr.shape, mask_arr.shape)

        # 保证image和mask第一维度为channel [c, w, h]
        # if img_arr.shape[1] != img_arr.shape[2]:
        #     img_arr = np.transpose(img_arr, (2, 0, 1))
        #     mask_arr = np.transpose(mask_arr, (2, 0, 1))

        # 特征提取的设置
        extractor = RadiomicsFeatureExtractor()

        if use_pyradiomics:
            extractor.enableImageTypeByName("Original")
            if min(mask_arr.shape) > 1: # 只对channel大于1的使用LoG滤波
                extractor.enableImageTypeByName("LoG", customArgs={"sigma": [1.0, 2.0, 3.0]})
            extractor.enableImageTypeByName("Wavelet")
            extractor.enableImageTypeByName("Square")
            extractor.enableImageTypeByName("SquareRoot")
            extractor.enableImageTypeByName("Logarithm")
            extractor.enableImageTypeByName("Exponential")
            kwards = dict()
            kwards["glcm"] = ["Autocorrelation", "JointAverage", "ClusterProminence", "ClusterShade",
                              "ClusterTendency", "Contrast", "Correlation", "DifferenceAverage",
                              "DifferenceEntropy", "DifferenceVariance", "JointEnergy", "JointEntropy",
                              "Imc1", "Imc2", "Idm", "Idmn", "Id", "Idn", "InverseVariance", "MaximumProbability",
                              "SumEntropy", "SumEntropy"]
            kwards["glrlm"] = []
            kwards["glszm"] = []
            kwards["gldm"] = []
            kwards["shape"] = []
            kwards["firstorder"] = []
            extractor.enableFeaturesByName(**kwards)
        else:
            settings = dict()
            settings['distances'] = [1]
            settings['binCount'] = 32
            settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
            settings['interpolator'] = sitk.sitkBSpline
            extractor.enableImageTypeByName("Original")
            extractor.disableAllFeatures()
            extractor.enableFeatureClassByName('firstorder')
            extractor.enableFeatureClassByName('shape')
            extractor.enableFeatureClassByName('glcm')
            if min(mask_arr.shape) > 1: # may raise FloatingPointError in 2D image
                extractor.enableFeatureClassByName('glrlm')

        # 去掉mask周围的0，得到一个缩小版的mask，再生成image，加速运算
        valid_range_z, valid_range_y, valid_range_x = get_compact_range(mask_arr)

        mask_arr = mask_arr[
                   valid_range_z[0]: valid_range_z[1] + 1,
                   valid_range_y[0]: valid_range_y[1] + 1,
                   valid_range_x[0]: valid_range_x[1] + 1]

        img_arr = img_arr[
                  valid_range_z[0]: valid_range_z[1] + 1,
                  valid_range_y[0]: valid_range_y[1] + 1,
                  valid_range_x[0]: valid_range_x[1] + 1]

        mask_itk = sitk.GetImageFromArray(mask_arr)

        # 保证image和mask的Spacing等信息一致
        img_itk = sitk.GetImageFromArray(img_arr)
        img_itk.SetSpacing(img.GetSpacing())
        img_itk.SetOrigin(img.GetOrigin())

        mask.SetSpacing(img.GetSpacing())
        mask.SetOrigin(img.GetOrigin())

        mask_itk.SetSpacing(img.GetSpacing())
        mask_itk.SetOrigin(img.GetOrigin())

        # 开始特征提取

        signature = extractor.execute(img_itk, mask_itk)

        # 特征提取完毕之后，根据ria还是pyradiomics设置不同的特征名称
        if use_pyradiomics:
            signature = OrderedDict(('_'.join(k.split('_')[1:] + k.split('_')[:1]), v) for k, v in signature.items())
        else:
            signature = OrderedDict((k, v[0] if isinstance(v, np.ndarray) else v) for k, v in signature.items())

        # 将特征更新+写入到临时文件里
        feature_vector.update(signature)
        with open(output_filename, 'w') as outputFile:
            writer = csv.DictWriter(outputFile, fieldnames=list(feature_vector.keys()), lineterminator='\n')
            writer.writeheader()
            writer.writerow(feature_vector)

        # Display message
        delta_t = datetime.now() - t

        # 为了避免打印不出来，使用error

        logging.getLogger('radiomics_s.batch').error('Case: %s %s %s processed in %s PID %s ',
                                                   case_id, case["image"], case["mask"],
                                                   delta_t, os.getpid())

    except KeyboardInterrupt:
        print('parent interrupted')

    except Exception:
        logging.getLogger('radiomics_s.batch').error('Feature extraction failed!', exc_info=True)

    return feature_vector


def main(data_csv, output_path, lib, cpus, img_reader):
    np.seterr(invalid='raise')
    # 2D 数据测试
    # parser.add_argument('--data_csv', help='image and mask columns', default='./example-2D/data.csv')
    # # parser.add_argument('--output', help='feature output folder', default='./example-2D/output/feature_ria.csv')
    # parser.add_argument('--output', help='feature output folder', default='./example-2D/output/feature_py.csv')
    # parser.add_argument('--lib', help='RIA or Pyradiomics', default='py')
    # parser.add_argument('--cpus', help='cpu cores', type=int, default=6)
    # parser.add_argument('--img_reader', help='dicom, nii, nrrd', default="nrrd")

    # 3D 数据测试
    # parser.add_argument('--data_csv', help='image and mask columns', default='./example-3D/data.csv')
    # parser.add_argument('--output', help='feature output folder', default='./example-3D/output/feature_ria.csv')
    # # parser.add_argument('--output', help='feature output folder', default='./example-3D/output/feature_py.csv')
    # parser.add_argument('--lib', help='RIA or Pyradiomics', default='ria')
    # parser.add_argument('--cpus', help='cpu cores', type=int, default=8)
    # parser.add_argument('--img_reader', help='dicom, nii, nrrd', default="dicom")

    # # 阜外 数据测试
    # parser.add_argument('--data_csv', help='image and mask columns', default='./example-fuwai/data.csv')
    # parser.add_argument('--output', help='feature output folder', default='./example-fuwai/output/feature_ria.csv')
    # # parser.add_argument('--output', help='feature output folder', default='./example-fuwai/output/feature_py.csv')
    # parser.add_argument('--lib', help='RIA or Pyradiomics', default='ria')
    # parser.add_argument('--cpus', help='cpu cores', type=int, default=8)
    # parser.add_argument('--img_reader', help='dicom, nii, nrrd', default="nrrd")

    ROOT = os.path.dirname(os.path.realpath(__file__))
    use_dicom = "dicom" in img_reader
    logging.getLogger('radiomics_s.batch').debug('Logging init')
    use_pyradiomics = lib in "pyradiomics"
    threading.current_thread().name = 'Main'
    REMOVE_TEMP_DIR = True
    NUM_OF_WORKERS = int(cpus)
    if NUM_OF_WORKERS < 1:
        NUM_OF_WORKERS = 1
    NUM_OF_WORKERS = min(cpu_count() - 1, NUM_OF_WORKERS)

    def term(sig_num, addtion):
        # os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
        print('Extraction abort, being killed by SIGTERM')
        pool.terminate()
        pool.join()
        return


    pool = Pool(NUM_OF_WORKERS)

    #print("Main PID {0}".format(os.getpid()))

    signal.signal(signal.SIGTERM, term)

    try:
        data_df = pd.read_csv(data_csv)
        try:
            images_list = data_df['image'].tolist()
        except:
            images_list = data_df['dataset'].tolist()
        masks_list = data_df['mask'].tolist()

        cases = [{
            'image': images_list[i],
            'mask': masks_list[i],
            'use_dicom': use_dicom,
            'img_reader': img_reader,
            'use_pyradiomics': use_pyradiomics
        } for i in range(len(images_list))]

        log = os.path.join(TEMP_DIR, 'log.txt')
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
        radio_logger = radiomics.logger
        log_handler = logging.FileHandler(filename=log, mode='a')
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter('%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s'))
        radio_logger.addHandler(log_handler)
        output_handler = radio_logger.handlers[0]  # Handler printing to the output
        output_handler.setFormatter(logging.Formatter('[%(asctime)-.19s] (%(threadName)s) %(name)s: %(message)s'))
        output_handler.setLevel(logging.INFO)  # Ensures that INFO messages are being passed to the filter
        output_handler.addFilter(info_filter('radiomics_s.batch'))
        logger = logging.getLogger('radiomics_s.batch')
        logger.info('Loaded %d jobs', len(cases))
        logger.info('Starting parralel pool with %d workers out of %d CPUs', NUM_OF_WORKERS, cpu_count())

        results = pool.map_async(extract_feature, cases).get(888888)
        c_f = output_path.split('/')[-1]
        if not os.path.exists(output_path.replace(c_f, '')):
            os.makedirs(output_path.replace(c_f, ''))
        try:
            # Store all results into 1 file
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=None)
            # with open(output_path, mode='w') as outputFile:
            #     writer = csv.DictWriter(outputFile,
            #                             fieldnames=list(results[0].keys()),
            #                             restval='',
            #                             extrasaction='raise',
            #                             lineterminator='\n')
            #     writer.writeheader()
            #     writer.writerows(results)

            if REMOVE_TEMP_DIR:
                logger.info('success')
                logger.info('Removing temporary directory %s (contains individual case results files)', TEMP_DIR)
                shutil.rmtree(TEMP_DIR)

        except Exception:
            logger.error('Error storing results into single file!', exc_info=True)

        chosen_feature = tools.choose_feature(output_path, use_pyradiomics=use_pyradiomics)
        chosen_feature.to_csv(output_path, index=False, encoding='utf-8')
    except (KeyboardInterrupt, SystemExit):
        print("...... Exit ......")
        pool.terminate()
        pool.join()
        return
    else:
        print("......end......")
        pool.close()

    print("System exit")
    pool.join()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--data_csv', help='image and mask columns', default='./example-debug/data.csv')
    # parser.add_argument('--output', help='feature output folder', default='./example-debug/output/feature_ria.csv')
    parser.add_argument('--output', help='feature output csv name', default='./example-debug/output/feature_py.csv')
    parser.add_argument('--lib', help='RIA or Pyradiomics', default='py')
    parser.add_argument('--cpus', help='cpu cores', type=int, default=8)
    parser.add_argument('--img_reader', help='dicom, nii, nrrd', default="dicom")
    args = parser.parse_args()
    main(args.data_csv, args.output, args.lib.lower(), args.cpus, args.img_reader)


