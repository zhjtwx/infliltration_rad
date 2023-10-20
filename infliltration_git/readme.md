## Introduction
This repository contains the main source code for our deep learning models, machine learning models and fusion models for predicting the invasiveness of pulmonary nodules., without model weights due to that it derived from a commercial software. The model is explained in the paper xxx.

### Prerequisites
- Ubuntu 16.04.4 LTS
- Python 3.6.13
- Pytorch 1.10.0+cu113
- NVIDIA GPU + CUDA_10.1 CuDNN_8.2
This repository has been tested on NVIDIA TITANXP. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

### Installation
Install dependencies:

```
pip install -r requirements.txt
```

#### Usage
- deep model：This model is mainly under the DL_ML folder. When training the two two-class models AAHAIS VS MIAIAC and AAHAISMIA VS IAC and the three-class model AAHAIS VS MIA VS IAC, you mainly need to modify the DL_ML/config_dl.py file. DL_ML/config_dl.py mainly modifies the training data and test data, as well as the hyperparameters of the model, etc.
    ```
    # train 
    python DL_ML/main_dfl_dl.py  DL_ML/config_dl.py
    ```
    During inference, you need to modify the ‘inference_mode’ in DL_ML/config_dl.py to ‘Ture’:
    ```
    # inference 
    python DL_ML/main_dfl_dl.py  DL_ML/config_dl.py
    ```
    #### The main parameters are as following:
    - --config: the path to the configuration file
    #### configuration file:
    - train task: DL_ML/config_dl.py
      - inference_mode = False
      - model_name = 'densenet36_fgpn'
      - train_set_dir: Save the image name and label csv file path of the training data
      - val_set_dirs: Save the image name and label csv file path of the val data
      - mode_save_base_dir: Model output address
    - infer task: DL_ML/config_dl.py
      - inference_mode = Ture
      - model_name = 'densenet36_fgpn'
      - save_csv: Result output

- ML model：This model is mainly under the ML folder. 
  - Radiomic feature extraction：
    ```
    # extraction 
    python extract.py --data_csv --output --lib --cpus
    ```
    Images are provided in two formats, dicom folder or nifti file, and mask only supports nifti format.

    - data_csv: Including two columns, image and mask, representing the paths of image and mask respectively.
      - image:dicom folder
      - mask:single file
    - output: The export file path of the feature, which is a csv file and will contain image and mask as image information.
    - lib: radiomics library, select RIA or Pyradiomics (case-insensitive, default is Pyradiomics)
    - cpus number of threads (number of CPU cores)
  - Feature selection:
    ```
    python feature_filter.py --feature_csv --target_csv --filters  --output_dir
    ```
    - feature_csv: feature csv file, all columns are features
    - target_csv: label file, only mask and label columns, label column is the label column
    - filters: filtering algorithm (ordered, comma separated), such as variance, kbest, lasso
    - output_dir: folder for output results
  - Machine learning：
    ```
    python learn.py --feature_csv --target_csv --tags_csv --models --output_dir
    ```
    - feature_csv: feature csv file, all columns except image and mask columns are features
    - target_csv: label file, only mask and label columns, label column is the label column
    - tags_csv: Tags to mark the training set or test set, with mask and dataset columns.
      - mask: mask file path
      - dataset: Data set, 0 is the training and verification set, 1 is the test set
  - Feature selection:
    ```
    python infer.py --feature_csv --model --label_encoder --feature_scalar --output
    ```
    - feature_csv: csv file, the first row is taken for prediction by default, including only the feature column
    - model: model file of joblib type
    - label_encoder: The encoder file of the category, used to convert the numerical type predicted by the model into the real category name
    - feature_scalar: normalized file
    - output: output category and probability, example: /output/predict.json

- fusion models(DL_ML model)：This model is mainly under the DL_ML folder. When training the two two-class models AAHAIS VS MIAIAC and AAHAISMIA VS IAC and the three-class model AAHAIS VS MIA VS IAC, you mainly need to modify the DL_ML/config_dl_rad.py file. DL_ML/config_dl_rad.py mainly modifies the training data and test data, as well as the hyperparameters of the model, etc.
    ```
    # train 
    python DL_ML/main_dfl_dl_rad.py  DL_ML/config_dl_rad.py
    ```
    During inference, you need to modify the ‘inference_mode’ in DL_ML/config_dl_rad.py to ‘Ture’:
    ```
    # inference 
    python DL_ML/main_dfl_dl_rad.py  DL_ML/config_dl_rad.py
    ```
    #### The main parameters are as following:
    - --config: the path to the configuration file
    #### configuration file:
    - train task: DL_ML/config_dl_rad.py
      - inference_mode = False
      - model_name = 'densenet36_fgpn_ml'
      - train_set_dir: Save the image name and label csv file path of the training data
      - train_rad_dir: Save the image name and radiomic feature csv file path of the training data
      - val_set_dirs: Save the image name, radiomic feature and label csv file path of the val data
      - mode_save_base_dir: Model output address
    - infer task: DL_ML/config_dl_rad.py
      - inference_mode = Ture
      - model_name = 'densenet36_fgpn_ml'
      - save_csv: Result output
