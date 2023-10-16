# -*- coding: UTF-8 -*-
from __future__ import print_function

import argparse
import copy
import glob
import json
import math
import os
import shutil
import traceback
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tools
import xgboost as xgb
from sklearn import datasets
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import check_is_fitted
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle


def _array_nan2zero(arr):
    for idx, item in enumerate(arr):
        if math.isnan(item) or math.isinf(item):
            arr[idx] = 0.
    return arr


def _ovresti_feature_importance(ovr):
    """
    class OneVSRest estimator get feature_importances_
    """
    check_is_fitted(ovr, 'estimators_')
    if not hasattr(ovr.estimators_[0], "feature_importances_"):
        raise AttributeError(
            "Base estimator doesn't have a feature_importances_ attribute.")
    coefs = [e.feature_importances_ for e in ovr.estimators_]
    if sp.issparse(coefs[0]):
        return sp.vstack(coefs)
    return np.vstack(coefs)


def _get_classifier(m, model_params, tv_data, is_binary, result, random_state=42, class_nb=None, auto_opt=False):
    tv_feature, tv_label = tv_data
    class_name = m
    if 'svc' == m:
        if auto_opt:
            param = dict(kernel=('linear', 'rbf'),
                         C=np.logspace(-4, 4, 20),
                         gamma=(1, 2, 3, 'auto'),
                         decision_function_shape=('ovo', 'ovr'),
                         shrinking=(True, False)
                         )
            clf_probe = RandomizedSearchCV(svm.SVC(), param, n_iter=200, scoring='accuracy', verbose=1,
                                           cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6,
                                           random_state=42)
            clf_probe.fit(tv_feature, tv_label)
            return svm.SVC(probability=True, random_state=random_state, **clf_probe.best_params_), class_name
        else:
            result["train"]["svc"]["parameter"] = model_params  # 01
            return svm.SVC(probability=True, **model_params), class_name
    elif 'nusvc' == m:
        if auto_opt:
            param = dict(kernel=('linear', 'rbf', 'linear'),
                         gamma=(1, 2, 3, 'auto'),
                         decision_function_shape=('ovo', 'ovr'),
                         shrinking=(True, False),
                         nu=(0.4, 0.5, 0.6)
                         )
            clf_probe = RandomizedSearchCV(svm.NuSVC(), param, n_iter=200, scoring='accuracy', verbose=1,
                                           cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6,
                                           random_state=42)
            clf_probe.fit(tv_feature, tv_label)
            return svm.NuSVC(probability=True, random_state=random_state, **clf_probe.best_params_), class_name
        else:
            result["train"]["nusvc"]["parameter"] = model_params  # 02
            return svm.NuSVC(probability=True, **model_params), class_name
    elif 'bayesgaussian' == m:
        # class_name = "Naive Bayes"
        return GaussianNB(**model_params), class_name
    elif 'bayesbernoulli' == m:
        return BernoulliNB(**model_params), class_name
    elif 'knn' == m:
        # class_name = "KNN"
        if auto_opt:
            param = dict(n_neighbors=tuple(range(class_nb, 4 * class_nb))[1::2],
                         weights=('uniform', 'distance'),
                         algorithm=('auto', 'ball_tree', 'kd_tree', 'brute'),
                         leaf_size=tuple(range(1, 3))
                         )
            clf_probe = GridSearchCV(KNeighborsClassifier(), param, scoring='accuracy', verbose=1,
                                     cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6)
            clf_probe.fit(tv_feature, tv_label)
            return KNeighborsClassifier(**clf_probe.best_params_), class_name
        else:
            if 'n_neighbors' in model_params:
                model_params['n_neighbors'] *= class_nb
                result["train"]["knn"]["parameter"] = model_params  # 03
            return KNeighborsClassifier(**model_params), class_name

    elif 'logistic' in m:
        # class_name = "Logistic Regression"
        if auto_opt:
            param = [{'penalty': ['l1', 'l2'],
                      'C': np.logspace(-4, 4, 20),
                      'solver': ['liblinear'],
                      'multi_class': ['ovr']},
                     {'penalty': ['l2'],
                      'C': np.logspace(-4, 4, 20),
                      'solver': ['lbfgs'],
                      'multi_class': ['ovr', 'multinomial']}]

            clf_probe = GridSearchCV(LogisticRegression(tol=1e-6, max_iter=1000), param, scoring='accuracy',
                                     verbose=1,
                                     cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6)
            clf_probe.fit(tv_feature, tv_label)
            return LogisticRegression(tol=1e-6, max_iter=1000, random_state=random_state,
                                      **clf_probe.best_params_), class_name
        else:
            result["train"]["logistic"]["parameter"] = model_params  # 04
            return LogisticRegression(**model_params), class_name

    elif 'decision' in m:
        # class_name = "Decision Tree"
        if auto_opt:
            param = dict(max_features=['auto', 'sqrt'],
                         max_depth=[int(x) for x in np.linspace(20, 200, 10)] + [None],
                         min_samples_split=[2, 5, 10],
                         min_samples_leaf=[1, 5, 10, 20, 50, 100]
                         )
            clf_probe = RandomizedSearchCV(DecisionTreeClassifier(), param, n_iter=200, scoring='accuracy',
                                           verbose=2,
                                           cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6)
            clf_probe.fit(tv_feature, tv_label)
            return DecisionTreeClassifier(random_state=random_state, **clf_probe.best_params_), class_name
        else:
            result["train"]["decision"]["parameter"] = model_params  # 05
            return DecisionTreeClassifier(random_state=random_state, **model_params), class_name

    elif 'random' in m:
        # class_name = "Random Forest"
        if auto_opt:
            param = dict(n_estimators=[int(x) for x in np.linspace(200, 1000, 10)],
                         max_features=['auto', 'sqrt'],
                         max_depth=[int(x) for x in np.linspace(10, 110, 11)] + [None],
                         min_samples_split=[2, 5, 10],
                         min_samples_leaf=[10, 20, 50, 100],
                         bootstrap=[True, False]
                         )
            clf_probe = RandomizedSearchCV(RandomForestClassifier(), param, n_iter=50, scoring='accuracy',
                                           verbose=2,
                                           cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6)
            clf_probe.fit(tv_feature, tv_label)
            return RandomForestClassifier(random_state=random_state, **clf_probe.best_params_), class_name
        else:
            return RandomForestClassifier(n_estimators=100, max_depth=100, random_state=random_state), class_name
    elif 'xgboost' in m:
        # class_name = "Gradient Boost Tree"
        if is_binary:
            return xgb.XGBClassifier(objective='binary:logistic', n_estimators=200, max_depth=10,
                                     random_state=random_state), class_name
        else:
            return xgb.XGBClassifier(objective='multi:softprob', n_estimators=200, max_depth=200,
                                     random_state=random_state, num_class=class_nb), class_name
    elif 'mlp' == m:
        # class_name = "Deep Learning"
        if auto_opt:
            param = dict(alpha=10.0 ** -np.arange(1, 10),
                         activation=['tanh', 'relu'],
                         solver=['adam'],
                         learning_rate=['adaptive']
                         )
            clf_probe = GridSearchCV(MLPClassifier(max_iter=5000, tol=1e-5), param, scoring='accuracy', verbose=1,
                                     cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6)
            clf_probe.fit(tv_feature, tv_label)
            return MLPClassifier(random_state=random_state, max_iter=5000, tol=1e-5,
                                 **clf_probe.best_params_), class_name
        else:
            result["train"]["mlp"]["parameter"] = model_params  # 06
            return MLPClassifier(random_state=random_state, max_iter=5000, tol=1e-5,
                                 **model_params), class_name

    elif 'adaboost' == m:
        return AdaBoostClassifier(n_estimators=200, random_state=random_state, **model_params), class_name
    elif 'lineardiscriminant' == m:
        return LinearDiscriminantAnalysis(**model_params), class_name
    elif 'quadraticdiscriminant' == m:
        return QuadraticDiscriminantAnalysis(**model_params), class_name
    elif 'sgd' == m:
        result["train"]["sgd"]["parameter"] = model_params  # 07
        return SGDClassifier(random_state=random_state, **model_params), class_name
    elif 'bagging' == m:
        return BaggingClassifier(random_state=random_state), class_name
    return None, None


def test_analyze(clf_name, clf, out_dir, df, df_learn, test_feature, encoded_test_label, label_encoder, is_binary,
                 origin_classes, result):
    predicts, proba, acc, class_report, auc_report, fpr, tpr, roc, samples, sen, spe = get_predict_report(clf,
                                                                                                          test_feature,
                                                                                                          encoded_test_label,
                                                                                                          label_encoder,
                                                                                                          is_binary,
                                                                                                          origin_classes,
                                                                                                          sess="test")
    stats = {}
    _get_roc_test(fpr, tpr, out_dir)
    tools.save_json({'acc': acc}, os.path.join(out_dir, 'acc.json'))
    result["test"][clf_name]["json"] = {}
    result["test"][clf_name]["json"]["acc"] = os.path.join(out_dir, 'acc.json')

    samples["mask"] = df[df_learn['dataset'] == 1]["mask"].tolist()
    samples["image"] = df[df_learn['dataset'] == 1]["image"].tolist()
    samples = samples[["image", "mask"] + [x for x in samples.columns if x not in ["image", "mask"]]]
    class_report.to_csv(os.path.join(out_dir, 'report.csv'), index=False, encoding='utf-8')
    test_samples_result_path = os.path.join(out_dir, 'test_samples_result.csv')
    samples.to_csv(test_samples_result_path, index=False, encoding='utf-8')

    tools.save_json(roc, os.path.join(out_dir, 'roc.json'))

    # save sen and spe
    np.savetxt(os.path.join(out_dir, 'sen.json'), sen)
    np.savetxt(os.path.join(out_dir, 'spe.json'), spe)

    # test analyze  ***
    result["test"][clf_name]["csv"] = {}
    result["test"][clf_name]["csv"]["report"] = os.path.join(out_dir, 'report.csv')
    result["test"][clf_name]["csv"]["samples_result"] = test_samples_result_path
    result["test"][clf_name]["json"]["roc"] = os.path.join(out_dir, 'roc.json')

    result["test"][clf_name]["json"]["sen"] = os.path.join(out_dir, 'sen.json')
    result["test"][clf_name]["json"]["spe"] = os.path.join(out_dir, 'spe.json')

    stats['roc'] = roc
    stats['accuracy'] = acc
    stats['sensitivity'] = tools.array2dict(origin_classes, sen)
    stats['specificity'] = tools.array2dict(origin_classes, spe)
    return stats


def _get_fold_folder(clf_name, fold, dataset=None, temp_dir=None):
    if dataset:
        return os.path.join(temp_dir, clf_name, 'fold_' + str(fold), dataset)
    else:
        return os.path.join(temp_dir, clf_name, 'fold_' + str(fold))


def _get_cv_folder(clf_name, dataset=None, temp_dir=None):
    if dataset:
        return os.path.join(temp_dir, clf_name, 'cv', dataset)
    else:
        return os.path.join(temp_dir, clf_name, 'cv')


def _get_roc_test(fpr, tpr, p):
    for origin_label in fpr:
        tools.roc_for_class([fpr[origin_label]],
                            [tpr[origin_label]],
                            class_name=origin_label,
                            save_path=os.path.join(p, 'roc_class_' + str(origin_label) + '.png'))


def _get_roc_train(clf_name, fold, d, fpr, tpr, p, result):
    result["train"][clf_name]["cross_valid"][fold][d]["image"] = {}
    # print(d)
    for origin_label in fpr:
        result["train"][clf_name]["cross_valid"][fold][d]["image"][origin_label] = {}
        tools.roc_for_class([fpr[origin_label]],
                            [tpr[origin_label]],
                            class_name=origin_label,
                            save_path=os.path.join(p, 'roc_class_' + str(origin_label) + '.png'))

        result["train"][clf_name]["cross_valid"][fold][d]["image"][origin_label]["roc_class"] = os.path.join(p,
                                                                                                             'roc_class_' + str(
                                                                                                                 origin_label) + '.png')


def get_predict_report(clf, x, y, label_encoder, is_binary, origin_classes, sess="train"):
    predicts = clf.predict(x)
    proba = clf.predict_proba(x)
    acc = clf.score(x, y)  #

    # class report
    if is_binary:
        report = classification_report(y,
                                       predicts,
                                       labels=label_encoder.transform(origin_classes),
                                       target_names=origin_classes,
                                       output_dict=True)
    else:  # multi class
        report = classification_report(np.argmax(y, axis=1),
                                       np.argmax(predicts, axis=1),
                                       labels=np.argmax(label_encoder.transform(origin_classes), axis=1),
                                       target_names=origin_classes,
                                       output_dict=True)
    report = pd.DataFrame.from_dict(report, orient='index')
    report['class'] = report.index
    report.drop('micro avg', inplace=True)
    report.drop('macro avg', inplace=True)

    # roc and auc for each class
    fpr, tpr, thresh, roc_auc, roc = dict(), dict(), dict(), dict(), dict()
    auc_report = []

    used_origin_idex = [0, 1] if is_binary else [i for i in sorted(list(set(list(np.argmax(y, axis=1)))))]

    Sensitivity = []
    Specificity = []
    for i in used_origin_idex:
        origin_label = origin_classes[i]
        if is_binary:
            class_specificity = report.loc[[l for l in origin_classes if l != origin_label][0]]['recall']
            fpr[origin_label], tpr[origin_label], thresh[origin_label] = roc_curve(y, proba[:, i], pos_label=i)
        else:
            class_specificity = 0.
            fpr[origin_label], tpr[origin_label], thresh[origin_label] = roc_curve(y[:, i], proba[:, i])

        thresh[origin_label][0] = 1.
        auc_score = auc(fpr[origin_label], tpr[origin_label])
        roc[origin_label] = dict()
        roc[origin_label]['fpr'] = _array_nan2zero(fpr[origin_label].tolist())
        roc[origin_label]['tpr'] = _array_nan2zero(tpr[origin_label].tolist())
        roc[origin_label]['thresh'] = list(map(np.float64, list(thresh[origin_label])))
        roc[origin_label]['auc'] = np.nan_to_num(auc_score)

        auc_report += [{
            'class': origin_label,
            'AUC': auc_score,
            'Sensitivity': report.loc[origin_label]['recall'],
            'Specificity': class_specificity
        }]

        Sensitivity.append(report.loc[origin_label]['recall'])
        Specificity.append(class_specificity)

    auc_report = pd.DataFrame(auc_report)
    auc_report = auc_report[['class'] + [c for c in auc_report.columns if c != 'class']]

    Sen = Sensitivity
    Spe = Specificity

    # samples
    indexs = np.arange(len(y))
    samples = pd.DataFrame(indexs, columns=['orders'])
    samples['predict'] = label_encoder.inverse_transform(predicts)
    samples['label'] = label_encoder.inverse_transform(y)
    samples['p_predicted'] = [max(p) for p in proba]
    for p in range(len(origin_classes)):
        samples.insert(loc=len(samples.columns),
                       column='p_' + str(origin_classes[p]),
                       value=proba[:, p])
    samples['correct'] = samples.apply(lambda row: str(row['predict']) == str(row['label']), axis=1)
    samples = samples[['label', 'predict', 'p_predicted'] +
                      ['p_' + str(origin_classes[p]) for p in range(len(origin_classes))] + ['correct']]

    return predicts, proba, acc, report, auc_report, fpr, tpr, roc, samples, Sen, Spe


def tv_analyze(clf, clf_name, x, y, fold, data_index, d, temp_dir, label_encoder, is_binary, origin_classes, immask,
               result):
    fold_path = _get_fold_folder(clf_name, fold, d, temp_dir=temp_dir)
    tools.makedir_ignore(fold_path)
    predicts, proba, acc, class_report, auc_report, fpr, tpr, roc, samples, sen, spe = get_predict_report(clf, x, y,
                                                                                                          label_encoder,
                                                                                                          is_binary,
                                                                                                          origin_classes)
    # add by tiansong for samples_prediction
    # immask is full image_path and mask_path for train+validation
    samples['mask'] = immask['mask'].values[data_index]
    samples["image"] = immask['image'].values[data_index]
    samples = samples[["image", "mask"] + [x for x in samples.columns if x not in ["image", "mask"]]]
    d_samples_result_path = os.path.join(fold_path, 'samples_result.csv')
    samples.to_csv(d_samples_result_path, index=False, encoding='utf-8')
    # end by tiansong
    _get_roc_train(clf_name, fold, d, fpr, tpr, fold_path, result)
    tools.save_json({'acc': acc}, os.path.join(fold_path, 'acc.json'))
    class_report.to_csv(os.path.join(fold_path, 'class_report.csv'), index=False, encoding='utf-8')
    auc_report.to_csv(os.path.join(fold_path, 'auc_report.csv'), index=False, encoding='utf-8')
    tools.save_json(roc, os.path.join(fold_path, 'roc.json'))

    np.savetxt(os.path.join(fold_path, 'x.csv'), x)
    np.savetxt(os.path.join(fold_path, 'y.csv'), y)

    np.savetxt(os.path.join(fold_path, 'sen.json'), sen)
    np.savetxt(os.path.join(fold_path, 'spe.json'), spe)

    result["train"][clf_name]["cross_valid"][fold][d]["json"] = {}
    result["train"][clf_name]["cross_valid"][fold][d]["csv"] = {}

    result["train"][clf_name]["cross_valid"][fold][d]["json"]["acc"] = os.path.join(fold_path, 'acc.json')
    result["train"][clf_name]["cross_valid"][fold][d]["json"]["roc"] = os.path.join(fold_path, 'roc.json')
    result["train"][clf_name]["cross_valid"][fold][d]["csv"]["class_report"] = os.path.join(fold_path,
                                                                                            'class_report.csv')
    result["train"][clf_name]["cross_valid"][fold][d]["csv"]["auc_report"] = os.path.join(fold_path, 'auc_report.csv')
    result["train"][clf_name]["cross_valid"][fold][d]["csv"]["x_data"] = os.path.join(fold_path, 'x.csv')
    result["train"][clf_name]["cross_valid"][fold][d]["csv"]["y_data"] = os.path.join(fold_path, 'y.csv')
    result["train"][clf_name]["cross_valid"][fold][d]["csv"]["samples_result"] = d_samples_result_path

    result["train"][clf_name]["cross_valid"][fold][d]["json"]["sen"] = os.path.join(fold_path, 'sen.json')
    result["train"][clf_name]["cross_valid"][fold][d]["json"]["spe"] = os.path.join(fold_path, 'spe.json')

    return fpr, tpr


def classification_train(clf, clf_name, summary_results, tv_feature, cv, tv_label, encoded_tv_label, output_path,
                         temp_dir, label_encoder, is_binary, origin_classes, immask, result):  # save
    cv_res = dict()
    # add by tiansong
    train_clf_stats = {}
    result["train"][clf_name]["cross_valid"] = {}
    sf = StratifiedKFold(n_splits=cv, random_state=88)
    for fold, c in enumerate(sf.split(tv_feature, tv_label)):
        result["train"][clf_name]["cross_valid"][fold] = {}
        t_index, v_index = c[0], c[1]
        tx, ty, vx, vy = tv_feature.iloc[t_index], encoded_tv_label[t_index], \
                         tv_feature.iloc[v_index], encoded_tv_label[v_index]
        fold_dataset_path = _get_fold_folder(clf_name, fold, temp_dir=temp_dir)
        tools.makedir_ignore(fold_dataset_path)

        # fit model
        clf.fit(tx, ty)
        clf_feature = ['decision', 'random']
        if clf_name in clf_feature:
            try:
                if is_binary:
                    feature_importance = np.expand_dims(clf.feature_importances_, axis=0)
                    fi_classes = origin_classes[1:].copy()
                else:
                    feature_importance = _ovresti_feature_importance(clf)
                    fi_classes = origin_classes.copy()
                fi_df = pd.DataFrame(feature_importance, columns=tv_feature.columns)
                fi_df.insert(0, 'class_names', fi_classes)
                feat_import_csv = os.path.join(fold_dataset_path, "feature_importance.csv")
                fi_df.to_csv(feat_import_csv, index=False)
                result["train"][clf_name]["cross_valid"][fold]["feature_importance"] = feat_import_csv
            except:
                traceback.print_exc()

        if clf_name == "logistic":
            try:
                feature_importance = clf.coef_
                if is_binary:
                    fi_classes = origin_classes[1:].copy()
                else:
                    fi_classes = origin_classes.copy()
                fi_df = pd.DataFrame(feature_importance, columns=tv_feature.columns)
                fi_df.insert(0, 'class_names', fi_classes)
                feat_import_csv = os.path.join(fold_dataset_path, "feature_importance.csv")
                fi_df.to_csv(feat_import_csv, index=False)
                result["train"][clf_name]["cross_valid"][fold]["feature_importance"] = feat_import_csv
            except:
                traceback.print_exc()
        # save model
        joblib.dump(clf, os.path.join(fold_dataset_path, 'model.joblib'))

        result["train"][clf_name]["cross_valid"][fold]["model"] = os.path.join(fold_dataset_path, 'model.joblib')

        cv_res[fold] = {'train': {'x': tx, 'y': ty, 'idx': t_index},
                        'valid': {'x': vx, 'y': vy, 'idx': v_index}}

        for d in cv_res[fold]:
            result["train"][clf_name]["cross_valid"][fold][d] = {}

            fpr, tpr = tv_analyze(clf, clf_name, cv_res[fold][d]['x'], cv_res[fold][d]['y'], fold=fold,
                                  data_index=cv_res[fold][d]['idx'], d=d, temp_dir=temp_dir,
                                  label_encoder=label_encoder, is_binary=is_binary, origin_classes=origin_classes,
                                  immask=immask, result=result)

            cv_res[fold][d]['roc'] = dict()
            for l in origin_classes:
                cv_res[fold][d]['roc'][l] = dict()
                cv_res[fold][d]['roc'][l]['fpr'] = fpr[l]
                cv_res[fold][d]['roc'][l]['tpr'] = tpr[l]

    result["train"][clf_name]["cv"] = {}
    acc_cv = {'acc': {'train': 1., 'valid': 1.}}
    for d in ['train', 'valid']:
        result["train"][clf_name]["cv"][d] = {}
        acc = []
        class_report_cv = []
        for f in range(cv):
            fold_dataset_path = _get_fold_folder(clf_name, f, d, temp_dir=temp_dir)
            auc_report = pd.read_csv(os.path.join(fold_dataset_path, 'auc_report.csv'))
            class_report = pd.read_csv(os.path.join(fold_dataset_path, 'class_report.csv'))
            auc_report['class'] = auc_report['class'].astype(str)
            class_report['class'] = class_report['class'].astype(str)
            report_df = auc_report.merge(class_report, on='class')
            ignore_columns = ['support']
            report_new_columns = list(report_df.columns)
            report_new_columns = [x for x in report_new_columns if x not in ignore_columns]
            report_df = report_df[report_new_columns]
            report_df.set_index('class', inplace=True)
            class_report_cv += [report_df]
            # acc
            with open(os.path.join(fold_dataset_path, 'acc.json')) as jf:
                acc += [json.load(jf)['acc']]

        acc_cv['acc'][d] = np.mean(np.array(acc))
        cv_dataset_path = _get_cv_folder(clf_name, d, temp_dir=temp_dir)
        tools.makedir_ignore(cv_dataset_path)
        report_mean = pd.concat(class_report_cv).groupby(level=0).mean()
        report_mean.to_csv(os.path.join(cv_dataset_path, 'report_mean.csv'), encoding='utf-8')
        result["train"][clf_name]["cv"][d]["report_mean.csv"] = os.path.join(cv_dataset_path, 'report_mean.csv')

    cv_path = _get_cv_folder(clf_name, temp_dir=temp_dir)
    pd.DataFrame.from_dict(acc_cv, orient='index').to_csv(os.path.join(cv_path, 'acc_mean.csv'), encoding='utf-8')
    result["train"][clf_name]["cv"]["image"] = {}
    # add by tiansong
    tv_roc = dict()
    for d in ['train', 'valid']:
        tv_roc[d] = defaultdict(dict)
    # end by tiansong
    for l in origin_classes:
        result["train"][clf_name]["cv"]["image"][l] = {}
        for d in ['train', 'valid']:
            result["train"][clf_name]["cv"]["image"][l][d] = {}
            fpr = [cv_res[k][d]['roc'][l]['fpr'] for k in cv_res.keys()]
            tpr = [cv_res[k][d]['roc'][l]['tpr'] for k in cv_res.keys()]
            # add by tiansong
            if len(fpr) != 1 or len(tpr) != 1:
                tv_roc[d][l][clf_name] = {'fpr': fpr[0], 'tpr': tpr[0]}
            else:
                raise ValueError(f'{clf_name} training fpr tpr len != 1')
            # end by tiansong
            tools.roc_for_cv(fpr, tpr, l,
                             os.path.join(_get_cv_folder(clf_name, d, temp_dir=temp_dir), str(l) + "_.png"), fold_nb=cv)

            result["train"][clf_name]["cv"]["image"][l][d]["roc"] = os.path.join(
                _get_cv_folder(clf_name, d, temp_dir=temp_dir), str(l) + "_.png")

    # update summary results by lvxiaogang
    train_clf_stats['roc'] = tv_roc
    result["train"][clf_name]["cv"]["csv"] = {}
    acc_path = os.path.join(temp_dir, clf_name, 'cv', 'acc_mean.csv')
    result["train"][clf_name]["cv"]["csv"]["acc_path"] = os.path.join(temp_dir, clf_name, 'cv', 'acc_mean.csv')

    acc_df = pd.read_csv(acc_path)
    summary_results += [{
        'model': clf_name,
        'train': acc_df['train'][0],
        'valid': acc_df['valid'][0]
    }]

    result['training_summary']['stat_compare']['accuracy'][clf_name] = {'train': acc_df['train'][0],
                                                                        'valid': acc_df['valid'][0]}

    model_compare_df = pd.DataFrame(summary_results)
    model_compare_df.to_csv(os.path.join(output_path, 'model_compare.csv'), index=False, encoding='utf-8')
    model_compare_df.to_csv(os.path.join(temp_dir, 'model_compare.csv'), index=False, encoding='utf-8')

    print(clf_name + " success")

    result["train"][clf_name]["csv"] = {}
    result["train"][clf_name]["csv"]["output_model_compare"] = os.path.join(output_path, 'model_compare.csv')
    result["train"][clf_name]["csv"]["temp_model_compare"] = os.path.join(temp_dir, 'model_compare.csv')

    return train_clf_stats


def learn(clf_names, models, models_params, summary_results, tv_feature, tv_label, is_binary, random_state, class_nb,
          auto_opt, encoded_tv_label, output_path, temp_dir, label_encoder, origin_classes, cv, immask, result):
    # initialize
    result['training_summary'] = dict()
    result['training_summary']['stat_compare'] = defaultdict(dict)
    result['training_summary']['roc_compare_fig'] = defaultdict(dict)
    result['training_summary']['roc_compare'] = defaultdict(dict)
    result['class_order'] = origin_classes.copy()
    roc_compare = None
    for m in models:
        clf, clf_name = _get_classifier(m, models_params[m], (tv_feature, tv_label), is_binary, result, random_state,
                                        class_nb=class_nb, auto_opt=auto_opt)
        clf_names += [clf_name]
        if not is_binary:
            clf = OneVsRestClassifier(clf)
        train_clf_stats = classification_train(clf, clf_name, summary_results, tv_feature, cv=cv, tv_label=tv_label,
                                               encoded_tv_label=encoded_tv_label, output_path=output_path,
                                               temp_dir=temp_dir,
                                               label_encoder=label_encoder, is_binary=is_binary,
                                               origin_classes=origin_classes,
                                               immask=immask,
                                               result=result)
        # add by tiansong roc
        if roc_compare is None:
            roc_compare = train_clf_stats['roc'].copy()
        else:
            for k1, v1 in roc_compare.items():
                for k2, v2 in v1.items():
                    v2.update(train_clf_stats['roc'][k1][k2].copy())
    result['classifier_order'] = clf_names.copy()
    # accuracy comparision
    accs = result['training_summary']['stat_compare']['accuracy']
    tv_acc_png = os.path.join(output_path, 'tv_acc_comparision.png')
    cates = ['train', 'valid']
    num_lists = []  # cates x xlabels
    xlabels = []  # cls_names
    for clf_name, v in accs.items():
        xlabels.append(clf_name)
        ys = []
        for c in cates:
            ys.append(v[c])
        num_lists.append(ys)
    num_lists = np.asarray(num_lists)
    num_lists = np.transpose(num_lists, (1, 0))
    tools.multibar_chart(num_lists.tolist(), cates, xlabels, title='', save_path=tv_acc_png, xlabel_rot=15)
    if os.path.isfile(tv_acc_png):
        result['training_summary']['stat_compare_fig'] = {'accuracy': tv_acc_png}
    # roc comparision
    for stage, v1 in roc_compare.items():
        for class_name, v2 in v1.items():
            fprs = []
            tprs = []
            c_names = []
            for c_name, v3 in v2.items():
                c_names.append(c_name)
                fprs.append(v3['fpr'])
                tprs.append(v3['tpr'])
            rocs_compare_png = os.path.join(output_path, f'{class_name}_{stage}_roc.png')
            auc_dict = tools.roc_for_clfs(fprs, tprs, c_names, f'{class_name} ROC', save_path=rocs_compare_png)
            if os.path.isfile(rocs_compare_png):
                result['training_summary']['roc_compare_fig'][class_name][stage] = rocs_compare_png
            result['training_summary']['roc_compare'][class_name][stage] = auc_dict


# test result
def testing(clf_names, output_path, temp_dir, label_encoder, df, df_learn, test_feature, encoded_test_label, is_binary,
            origin_classes, result):
    rocs_compare = defaultdict(dict)
    stats_compare = defaultdict(dict)
    result['testing_summary'] = defaultdict(dict)
    result['training_summary']['feature_importance'] = defaultdict(dict)
    for clf_name in clf_names:
        res_path = os.path.join(output_path, "models", clf_name)
        tools.makedir_delete(res_path)
        best_fold = None
        best_indicator = 0
        for cr in glob.glob(os.path.join(temp_dir, clf_name) + '/fold_*/valid/acc.json'):
            acc_score = tools.load_json(cr)["acc"]
            if acc_score >= best_indicator:
                best_indicator = acc_score
                best_fold = int(cr.split('fold_')[1].split('/')[0])

        model_path = _get_fold_folder(clf_name, best_fold, 'model.joblib', temp_dir=temp_dir)
        shutil.copy(model_path, res_path + '/model.joblib')
        # add by tiansong feature_importance
        src_feat_import_path = _get_fold_folder(clf_name, best_fold, 'feature_importance.csv', temp_dir=temp_dir)
        if os.path.isfile(src_feat_import_path):
            dest_feat_import_path = os.path.join(res_path, 'feature_importance.csv')
            shutil.copy(src_feat_import_path, dest_feat_import_path)
            result['training_summary']['feature_importance'][clf_name] = dest_feat_import_path
        # tv_samples_result
        for d in ['train', 'valid']:
            samples_result_srcpath = _get_fold_folder(clf_name, best_fold, os.path.join(d, 'samples_result.csv'),
                                                      temp_dir=temp_dir)
            samples_result_destpath = os.path.join(res_path, f"{d}_samples_result.csv")
            shutil.copy(samples_result_srcpath, samples_result_destpath)
            result['train'][clf_name]['csv'][f'{d}_samples_result'] = samples_result_destpath
        # end by tiansong
        np.save(res_path + '/encoder.npy', label_encoder.classes_)
        model = joblib.load(model_path)
        clf_stats = test_analyze(clf_name, model, res_path, df, df_learn, test_feature, encoded_test_label,
                                 label_encoder, is_binary,
                                 origin_classes, result)
        for cate in clf_stats['roc'].keys():
            rocs_compare[cate][clf_name] = clf_stats['roc'][cate]
        for k, v in clf_stats.items():
            if k == 'roc':
                continue
            stats_compare[k][clf_name] = v
    for c, v in rocs_compare.items():
        fprs = []
        tprs = []
        names = []
        for subclf, tfpr in v.items():
            fprs.append(tfpr['fpr'])
            tprs.append(tfpr['tpr'])
            names.append(subclf)
        rocs_compare_png = os.path.join(output_path, '{}_test_roc_compare.png'.format(c))
        auc_dict = tools.roc_for_clfs(fprs, tprs, names, c, save_path=rocs_compare_png)
        result['testing_summary']['roc_compare_fig'][c] = rocs_compare_png
        result['testing_summary']['roc_compare'][c] = auc_dict
    for stname, v in stats_compare.items():
        xs = []
        ys = []
        for k, subv in v.items():
            xs.append(k)
            ys.append(subv)
        if stname in ['accuracy', ]:
            result['testing_summary']['stat_compare'][stname] = tools.array2dict(clf_names, ys)
        else:
            result['testing_summary']['stat_compare_by_class'][stname] = tools.dict_layer_switch(
                tools.array2dict(clf_names, ys))
    st_order = []
    num_lists = []
    for stname, v in result['testing_summary']['stat_compare'].items():
        st_order.append(stname)
        vs = []
        for k1 in clf_names:
            vs.append(v[k1])
        num_lists.append(vs)

    num_lists = np.asarray(num_lists)
    num_lists = np.transpose(num_lists, (1, 0))
    stat_compare_png = os.path.join(output_path, 'testing_stat_compare.png')
    tools.multibar_chart(num_lists.tolist(), clf_names, st_order, title='', save_path=stat_compare_png, xlabel_rot=15)
    if os.path.isfile(stat_compare_png):
        result['testing_summary']['stat_compare_fig'] = {'accuracy': stat_compare_png}
    # TODO: specificity sensitivity compare by classes


def main(feature_path, target_path, tags_path, models, output_path, cv, auto_opt, cls_params):
    param_dict = {
        'svc': {},
        'nusvc': {},
        'bayesgaussian': {},
        'bayesbernoulli': {},
        'knn': {},
        'logistic': {},
        'decision': {},
        'random': {},
        'xgboost': {},
        'mlp': {},
        'adaboost': {},
        'lineardiscriminant': {},
        'quadraticdiscriminant': {},
        'sgd': {},
        'bagging': {},
    }

    result = {
        'classes': ['svc', 'nusvc', 'bayesgaussian', 'bayesbernoulli', 'knn', 'logistic', 'decision', 'random',
                    'xgboost', 'mlp', 'adaboost', 'lineardiscriminant', 'quadraticdiscriminant', 'sgd', 'bagging'],
        'train': param_dict,
        'test': copy.deepcopy(param_dict),
    }

    clf_names = []  # formal name
    warnings.filterwarnings('ignore')

    # TEMP_DIR = tempfile.mkdtemp()
    TEMP_DIR = os.path.join(output_path, "temp")
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    else:
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)

    df = pd.read_csv(feature_path)
    df = shuffle(df, random_state=88)
    target = pd.read_csv(target_path)
    tags = pd.read_csv(tags_path)

    df, target, tags = tools.prepare_feature_n_label(df, target, tags)

    # 构造数据集
    feature_columns = [x for x in df.columns if x not in tools.keywords]
    df_learn = df[feature_columns]
    df_learn = tools.preprocessing(df_learn)

    # 归一化
    std_scaler = StandardScaler()
    df_learn[df_learn.columns] = std_scaler.fit_transform(df_learn)
    joblib.dump(std_scaler, os.path.join(output_path, "scalar.joblib"))

    df_learn['label'] = target['label'].tolist()
    df_learn['dataset'] = tags['dataset'].tolist()
    df_learn = df_learn[['label', 'dataset'] + feature_columns]

    # 训练集
    tv_df = df_learn[df_learn['dataset'] == 0]
    tv_label = [str(l) for l in tv_df.label.to_list()]
    tv_feature = tv_df[feature_columns]
    # add by tiansong
    tv_immask = {}
    tv_immask['image'] = df[df_learn['dataset'] == 0]['image']
    tv_immask['mask'] = df[df_learn['dataset'] == 0]['mask']
    # end by tiansong

    # 测试集
    test_df = df_learn[df_learn['dataset'] == 1]
    test_label = [str(l) for l in test_df.label.to_list()]
    test_feature = test_df[feature_columns]

    class_nb = len(set(df_learn.label.tolist()))
    is_binary = class_nb == 2

    # Iris 数据集，用于baseline测试
    enable_test = False
    if enable_test:
        iris = datasets.load_iris()
        iris_X, iris_y = iris.data, iris.target
        columns = [str(s) for s in range(iris_X.shape[1])]
        skf = StratifiedKFold(n_splits=5).split(iris_X, iris_y)
        iris_tv, iris_test = skf.next()
        tv_df = pd.DataFrame(iris_X[iris_tv], columns=columns)
        test_df = pd.DataFrame(iris_X[iris_test], columns=columns)
        class_nb = len(set(iris_y))
        tv_label = iris_y[iris_tv]
        test_label = iris_y[iris_test]
        tv_feature = tv_df[[x for x in columns if x not in tools.keywords]]
        test_feature = test_df[[x for x in columns if x not in tools.keywords]]

    # label encoding
    label_encoder, encoded_label, _ = tools.encode_b(tv_label + test_label) if not is_binary else tools.encode_l(
        tv_label + test_label)
    encoded_tv_label = encoded_label[0: len(tv_label)]
    encoded_test_label = encoded_label[len(tv_label):]

    origin_classes = list(label_encoder.classes_)
    origin_classes_tv = sorted(list(set(tv_label)))
    origin_classes_test = sorted(list(set(test_label)))

    # 结果总表
    summary_results = []
    random_state = 42

    learn(clf_names, models, cls_params, summary_results, tv_feature, tv_label,
          is_binary=is_binary, random_state=random_state,
          class_nb=class_nb, auto_opt=auto_opt,
          encoded_tv_label=encoded_tv_label, output_path=output_path,
          temp_dir=TEMP_DIR, label_encoder=label_encoder,
          origin_classes=origin_classes, cv=cv, immask=tv_immask, result=result)
    # add by tiansong
    if test_feature.shape[0] > 0:
        testing(clf_names, output_path=output_path, temp_dir=TEMP_DIR, label_encoder=label_encoder, df=df,
                df_learn=df_learn, test_feature=test_feature, encoded_test_label=encoded_test_label, is_binary=is_binary,
                origin_classes=origin_classes, result=result)
    else:
       print(f"No testing dataset {test_feature.shape}")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # fuwai
    # parser.add_argument('--feature_csv', help='feature csv file', default='./example-fuwai/output/filter/feature_selected.csv')
    # parser.add_argument('--target_csv', help='target csv file', default='./example-fuwai/target.csv')
    # parser.add_argument('--tags_csv', help='tags', default='./example-fuwai/tags.csv')
    # parser.add_argument('--output_dir', help='output csv file', default='./example-fuwai/output/')

    # debug
    parser.add_argument('--feature_csv', help='feature csv file', default='./example-debug/output/feature_selected.csv')
    parser.add_argument('--target_csv', help='target csv file', default='./example-debug/target.csv')
    parser.add_argument('--tags_csv', help='tags', default='./example-debug/tags.csv')
    parser.add_argument('--output_dir', help='output csv file', default='./example-debug/output')

    # 3D 模型测试
    # parser.add_argument('--feature_csv', help='feature csv file', default='./example-3D/output/filter/feature_selected.csv')
    # parser.add_argument('--target_csv', help='target csv file', default='./example-3D/label_N_4.csv')
    # parser.add_argument('--tags_csv', help='tags', default='./example-3D/tags.csv')
    # parser.add_argument('--output_dir', help='output csv file', default='./example-3D/output/')

    # 2D 模型测试
    # parser.add_argument('--feature_csv', help='feature csv file', default='./example-2D/output/filter/feature_selected.csv')
    # parser.add_argument('--target_csv', help='target csv file', default='./example-2D/target2.csv')
    # parser.add_argument('--tags_csv', help='tags', default='./example-2D/tags2.csv')
    # parser.add_argument('--output_dir', help='output csv file', default='./example-2D/output/learn')

    # parser.add_argument('--models', help='models', default='knn')
    # parser.add_argument('--models', help='models',
    #                     default='knn, bayes, xgboost, deep, svm, logistic, decision_tree, random_forest')
    parser.add_argument('--cv', help='number of cross validation', type=int, default=5)
    parser.add_argument('--auto_opt', help="auto optimization", action='store_true', default=False)
    parser.add_argument('--cls_params', help="classifier parameters in json format", default=None, type=json.loads)

    args = parser.parse_args()

    # debug
    cls_params = args.cls_params
    if cls_params is None:
        raise ValueError("cls_params is None")

    models = list(cls_params.keys())
    result = main(args.feature_csv, args.target_csv, args.tags_csv, models, args.output_dir, args.cv,
                  args.auto_opt, cls_params)

    with open(os.path.join(args.output_dir, "class_result.json"), 'w') as f:
        json.dump(result, f, indent=4)
