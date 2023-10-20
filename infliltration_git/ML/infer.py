# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import argparse
import tools
from sklearn.externals import joblib


# 2D数据参数测试
# parser.add_argument('--feature_csv', help='feature csv file, will take the first row as feature', default='./example-2D/output/filter/feature_selected.csv')
# parser.add_argument('--model', help='model file, .joblib', default='./example-2D/output/learn/models/xgboost/model.joblib')
# parser.add_argument('--label_encoder', help='json', default='./example-2D/output/learn/models/xgboost/encoder.npy')
# parser.add_argument('--feature_scalar', help='json', default='./example-2D/output/learn/scalar.joblib')
# parser.add_argument('--output', help='output csv file', default='./example-2D/output/infer/predict.json')

# 3D数据参数测试
# parser.add_argument('--feature_csv', help='feature csv file, will take the first row as feature', default='./example-3D/output/filter/feature_selected.csv')
# parser.add_argument('--model', help='model file, .joblib', default='./example-3D/output/models/SVM/model.joblib')
# parser.add_argument('--label_encoder', help='json', default='./example-3D/output/models/SVM/encoder.npy')
# parser.add_argument('--feature_scalar', help='json', default='./example-3D/output/scalar.joblib')
# parser.add_argument('--output', help='output csv file', default='./example-3D/predict.json')


def main(df_path, model_path, output_path, encoder_path, scalar_path):
    data = pd.read_csv(df_path)
    data = data.iloc[0:1, :]
    data = data[[x for x in data.columns if x not in tools.keywords]]
    data = tools.preprocessing(data)
    scalar = joblib.load(scalar_path)
    try:
        data[data.columns] = scalar.transform(data)
    except ValueError:
        raise ValueError("输入特征需要和训练模型时选择的特征一致")

    clf = joblib.load(model_path)

    #
    # def debug():
    #     tags = pd.read_csv("./example-debug/tags.csv")
    #     target = pd.read_csv("./example-debug/target.csv")
    #     gt = pd.merge(tags, target, on=["image", "mask"])
    #     for idx, row in data.iterrows():
    #         res = dict(zip(list(np.load(encoder_path)), [float(p) for p in list(clf.predict_proba(data.iloc[idx:idx+1, :])[0])]))
    #         print(res, gt.iloc[idx:idx+1, :]["label"].tolist()[0])
    #     print()
    #
    # debug()


    x = data.iloc[0:1, :]
    res = dict(zip(list(np.load(encoder_path)), [float(p) for p in list(clf.predict_proba(x)[0])]))
    print(res)
    tools.save_json(res, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--feature_csv', help='feature csv file, will take the first row as feature', default='./example-debug/output/feature_selected.csv')
    parser.add_argument('--model', help='model file, .joblib', default='./example-debug/output/models/knn/model.joblib')
    parser.add_argument('--label_encoder', help='json', default='./example-debug/output/models/xgboost/encoder.npy')
    parser.add_argument('--feature_scalar', help='json', default='./example-debug/output/scalar.joblib')
    parser.add_argument('--output', help='output csv file', default='./example-debug/output/infer/predict.json')
    args = parser.parse_args()
    main(args.feature_csv, args.model, args.output, args.label_encoder, args.feature_scalar)
