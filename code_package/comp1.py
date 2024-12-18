# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/12/22
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import copy
import math
import os
import sys
import traceback
from collections import defaultdict
from typing import List, Union, Iterable

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.api as sm
from IPython.core.display import display
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib.markers import MarkerStyle
# import matplotlib
# matplotlib.use('Agg')
from matplotlib.pyplot import MultipleLocator
from pandas import DataFrame
from scipy.stats import ttest_1samp
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, AdaBoostClassifier, \
    AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import LassoCV, LogisticRegression, ElasticNetCV, MultiTaskElasticNetCV, MultiTaskLassoCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from onekey_algo import get_param_in_cwd, init_CN
from onekey_algo.custom.components.delong import calc_95_CI
from onekey_algo.custom.components.metrics import analysis_pred_binary
from onekey_algo.utils import create_dir_if_not_exists
from onekey_algo.utils.about_log import logger
import matplotlib.colors as mcolors


def read_data(filename: str):
    assert os.path.exists(filename), f'{filename} 不存在！'
    if filename.endswith('.csv'):
        return pd.read_csv(filename, header=0)
    elif filename.endswith('.xlsx'):
        pd.read_excel(filename, header=0)
    else:
        raise ValueError(f'文件类型未定义！')


def compress_feature(features: np.array, dim: int) -> np.array:
    assert len(features.shape) == 2, '特征的维度必须为2维'
    if dim > features.shape[1]:
        logger.warning(f"降维的维度（{dim}）不能多于样本数({features.shape[0]}), 使用{features.shape[1]}作为降维维度！")
        dim = features.shape[1]
    pca = PCA(dim, random_state=0)
    features = pca.fit_transform(features)
    return features


def compress_df_feature(features: pd.DataFrame, dim: int, not_compress: Union[str, List[str]] = None,
                        prefix='') -> pd.DataFrame:
    """
    压缩深度学习特征
    Args:
        features: 特征DataFrame
        dim: 需要压缩到的维度，此值需要小于样本数
        not_compress: 不进行压缩的列。
        prefix: 所有特征的前缀。

    Returns:

    """
    if not_compress is not None:
        if isinstance(not_compress, str):
            not_compress = [not_compress]
        elif not isinstance(not_compress, Iterable):
            raise ValueError(f"not_compress设置出错！")
        not_compress_data = features[not_compress]
        features = features[[c for c in features.columns if c not in not_compress]]
        features = compress_feature(features, dim=dim)
        features = pd.DataFrame(features, columns=[f"{prefix}{i}" for i in range(features.shape[1])])
        return pd.concat([not_compress_data, features], axis=1)
    else:
        features = compress_feature(features, dim=dim)
        features = pd.DataFrame(features, columns=[f"{prefix}{i}" for i in range(features.shape[1])])
        return features


def split_dataset(X_data, y_data, test_size=0.2, random_state=0, reset_index: bool = True):
    def __reset_index(d_):
        d_ = d_.reset_index()
        d_ = d_.drop(['index'], axis=1)
        return d_

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=random_state, test_size=test_size)
    if reset_index:
        X_train = __reset_index(X_train)
        X_test = __reset_index(X_test)
        y_train = __reset_index(y_train)
        y_test = __reset_index(y_test)
    return X_train, X_test, y_train, y_test


def cluster_analysis(features, n_clusters=2):
    cluster = KMeans(n_clusters=n_clusters, random_state=0)
    cluster.fit(features)
    return cluster


def normalize_df(data: pd.DataFrame, not_norm: Union[str, List[str]] = None,
                 method='gaussian', group: str = None, sub_group: Union[List, List[str]] = None,
                 use_train: bool = False, verbose=False) -> pd.DataFrame:
    """
    Normalize data frame
    Args:
        data: DataFrame to be Normalized.
        not_norm: Columns not be normalized.
        method: method to be used, choice: gaussian, min_max, default gaussian.
        group: Normalize by group separately, default None treated as a whole.
        sub_group: sub group to normalize.
        use_train: bool, 是否使用训练集的指标对其他group进行标准化。
        verbose: verbose

    Returns: Normalized Dataframe

    """
    if not_norm is None:
        not_norm = []
    elif isinstance(not_norm, str):
        not_norm = [not_norm]
    if group is not None:
        not_norm = not_norm + [group]
        assert group in data.columns, f"group: {group} 分组没有在{data.columns}!"
    columns = [c for c in data.columns if c not in not_norm]
    new_data = data.copy(deep=True)

    def _norm_data(data_, spec_desc=None):
        desc = data_.describe()
        for column in columns:
            try:
                mean = (spec_desc is not None and spec_desc[column]['mean']) or desc[column]['mean']
                std = (spec_desc is not None and spec_desc[column]['std']) or desc[column]['std']
                min_ = (spec_desc is not None and spec_desc[column]['min']) or desc[column]['min']
                max_ = (spec_desc is not None and spec_desc[column]['max']) or desc[column]['max']
                if method == 'gaussian':
                    data_[column] = (data_[column] - mean) / std
                else:
                    data_[column] = (data_[column] - min_) / (max_ - min_)
            except Exception as e:
                if verbose:
                    traceback.print_exc()
                logger.warning(f"特征：{column}存在问题：{e}，造成z-score失败！")
        return data_

    if group is None:
        _norm_data(new_data)
    else:
        ug_list = []
        ugroups = list(np.unique(data[group]))
        if 'train' in ugroups and use_train:
            logger.info('正在使用训练集预定数据进行标准化。')
            train_desc = new_data[new_data[group] == 'train'].describe()
        else:
            train_desc = None
        for ug in ugroups:
            ug_data = new_data[new_data[group] == ug]
            if sub_group is None or ug in sub_group:
                _norm_data(ug_data, spec_desc=train_desc)
            ug_list.append(ug_data)
        new_data = pd.concat(ug_list, axis=0)
    return new_data


def convert2onehot(data, n_classes):
    data = np.reshape(data, -1)
    onehot_encoder = []
    for d in data:
        onehot = [0] * n_classes
        onehot[d] = 1
        onehot_encoder.append(onehot)
    return np.array(onehot_encoder)


def draw_roc_per_class(y_test, y_score, n_classes, title='ROC per Class', include_spec_class: bool = True,
                       mapping=None, subset=None, use_youden: bool = False):
    """

    Args:
        mapping: label的映射
        y_test: 真实标签
        y_score: 预测标签
        n_classes: 类别数
        title: 标题
        include_spec_class: 是否包括每个细分标签的ROC曲线。
        subset: subset
        use_youden: 是否使用youden index重算sens spec。

    Returns:

    """
    precision = get_param_in_cwd('display.precision', 3)
    if mapping is None:
        mapping = {}
    y_test_binary = convert2onehot(y_test, n_classes=n_classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    metric_results = []
    # Compute micro-average ROC curve and ROC area
    # print(y_test_binary.ravel().shape, y_score.ravel().shape)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binary.ravel(), y_score.ravel())
    acc, auc_, ci, tpr_, tnr, ppv, npv, pre, re, f1, thres = analysis_pred_binary(y_test_binary.ravel(),
                                                                                  y_score.ravel(),
                                                                                  use_youden=use_youden)

    roc_auc["micro"] = auc_
    ci = f"{ci[0]:.3f}-{ci[1]:.3f}"
    metric_results.append(['Micro-AUC', acc, auc_, ci, tpr_, tnr, ppv, npv, pre, re, f1, thres, subset])
    micro_str = f"micro AUC: {{auc:.{precision}f}} (95%CI {ci})"
    lw = 2
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=micro_str.format(auc=roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    try:
        ci_dict = []
        for i in range(n_classes):
            try:
                fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_score[:, i])
                acc, auc_, ci, tpr_, tnr, ppv, npv, pre, re, f1, thres = analysis_pred_binary(y_test_binary[:, i],
                                                                                              y_score[:, i])
                ci_dict.append(ci)
                ci = f"{ci[0]:.3f}-{ci[1]:.3f}"
                metric_results.append([mapping[i] if i in mapping else i,
                                       acc, auc_, ci, tpr_, tnr, ppv, npv, pre, re, f1, thres, subset])
                roc_auc[i] = auc_
            except Exception as e:
                logger.error(f'解析{i}类别出错, {e}')
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        mean_ci = np.mean(ci_dict, axis=0)
        ci = f"{mean_ci[0]:.3f}-{mean_ci[1]:.3f}"
        macro_str = f"macro AUC: {{auc:.{precision}f}} (95%CI {ci})"
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=macro_str.format(auc=roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )
        per_class_str = f"{{cn}} AUC: {{auc:.{precision}f}} (95%CI {{ci}})"
        if include_spec_class:
            for i in range(n_classes):
                ci = ci_dict[i]
                ci = f"{ci[0]:.3f}-{ci[1]:.3f}"
                plt.plot(
                    fpr[i],
                    tpr[i],
                    lw=lw,
                    label=per_class_str.format(cn=mapping[i] if i in mapping else f"class {i}", auc=roc_auc[i], ci=ci),
                )
    except:
        logger.error(f'解析每个类别的ROC出错，大概率是应为数据没有指定类别的样本！')
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    metric_results = pd.DataFrame(metric_results, columns=['ModelName', 'Acc', 'AUC', '95% CI', 'Sensitivity',
                                                           'Specificity', 'PPV', 'NPV', 'Precision', 'Recall', 'F1',
                                                           'Threshold', 'Cohort'])
    return metric_results


def draw_roc(y_test, y_score, title='ROC', labels=None, auto_point: bool = False, **kwargs):
    """
    绘制ROC曲线
    Args:
        y_test: list或者array，为真实结果。
        y_score: list orray或者array，为模型预测结果。
        title: 图标题
        labels: 图例名称
        auto_point: 是否自动检测是否为人工标注。

    Returns:

    """
    precision = get_param_in_cwd('display.precision', 3)
    if not isinstance(y_test, (list, tuple)):
        y_test = [y_test]
    if not isinstance(y_score, (list, tuple)):
        y_score = [y_score]
    if labels is None:
        labels = [''] * len(y_score)
    if not (len(y_test) == len(y_score) == len(labels)):
        logger.warning(f'输入数据量不相等，我们将忽略多余的数据！')
    aux_colors, _ = zip(*sorted(mcolors.cnames.items(), key=lambda x: x[1], reverse=False))
    colors = ["deeppink", "navy", "aqua", "darkorange", "cornflowerblue"] + list(aux_colors)
    ls = kwargs.get('ls', ['-', ':', '-', ':'])
    point_colors = kwargs.get('point_colors', None) or colors
    markers = kwargs.get('markers', None) or list(MarkerStyle.markers.keys())
    point_idx = 0
    for idx, (y_test_, y_score_, label) in enumerate(zip(y_test, y_score, labels)):
        y_score_ = np.squeeze(np.array(y_score_))
        # enc = OneHotEncoder(handle_unknown='ignore')
        # y_test_binary = enc.fit_transform(y_test_.reshape(-1, 1)).toarray()
        if len(y_score_.shape) == 1:
            y_score_1 = y_score_
        else:
            y_score_1 = y_score_[:, 1]
        fpr, tpr, _ = roc_curve(y_test_, y_score_1)
        auc_, ci = calc_95_CI(np.squeeze(y_test_), y_score_1)
        roc_auc = auc(fpr, tpr)
        label_info = f"{label} AUC: {{roc_auc:.{precision}f}} (95%CI {{lo:.{precision}f}}-{{up:.{precision}f}})"
        label_info = label_info.format(roc_auc=roc_auc, lo=ci[0], up=ci[1])
        if len(fpr) == 3 and len(tpr) == 3 and auto_point:
            plt.scatter(fpr[1], tpr[1], label=label_info, linewidths=get_param_in_cwd('lines.linewidth', 4),
                        color=point_colors[point_idx % len(point_colors)], marker=markers[point_idx % len(markers)])
            point_idx += 1
        else:
            plt.plot(fpr, tpr, label=label_info,
                     color=colors[idx % len(colors)], linestyle=ls[idx % len(ls)],
                     linewidth=get_param_in_cwd('lines.linewidth', 4))

    plt.plot([0, 1], [0, 1], "k--", lw=get_param_in_cwd('lines.linewidth', 2))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")


def draw_calibration(y_test, pred_scores, model_names, remap: bool = False, **kwargs):
    """

    Args:
        y_test:
        pred_scores:
        model_names:
        remap: 是否使用逻辑回归重新映射。
        **kwargs:

    Returns:

    """
    version_info = sys.version_info
    assert version_info.major >= 3 and version_info.minor > 6, "Python版本必须3.7及以上。"
    if not isinstance(y_test, (tuple, list)):
        y_test = [y_test] * len(model_names)
    if len(pred_scores) != len(model_names):
        logger.warning(f'输入数据量不相等，我们将忽略多余的数据！')
    from sklearn.calibration import CalibrationDisplay
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 1)
    fig = plt.figure(figsize=(10, 10))
    ax_calibration_curve = fig.add_subplot(gs[0, 0])

    def moving_average(interval, windowsize):
        window = np.ones(int(windowsize)) / float(windowsize)
        return np.convolve(interval, window, 'full')

    for model_name, scores, gt in zip(model_names, pred_scores, y_test):
        if remap:
            if kwargs.get('EX', None) is None:
                model = LogisticRegression()
            else:
                model = RandomForestClassifier(**kwargs.get('EX'))
            if len(scores.shape) == 1:
                scores = np.reshape(np.array(scores), [-1, 1])
                model.fit(scores, gt)
            else:
                model.fit(scores, gt)
            scores = model.predict_proba(scores)

            #             display(pd.DataFrame(scores))
            scores = np.array(normalize_df(pd.DataFrame(scores), method='minmax'))
        if kwargs.get('smooth', False):
            if len(scores.shape) == 1:
                scores = scipy.signal.savgol_filter(scores, kwargs.get('window_length', 9), kwargs.get('k', 3))
            else:
                scores = scipy.signal.savgol_filter(scores[:, 1], kwargs.get('window_length', 9), kwargs.get('k', 3))
            scores = np.clip(scores, 0, 1)
        if len(scores.shape) == 1:
            pred = scores
        else:
            pred = scores[:, 1]
        # print(gt.shape, pred.shape, np.reshape([0], (-1, 1)).shape)
        # gt = np.concatenate([np.reshape([0] * 20, (-1, 1)), gt], axis=0)
        # pred = np.concatenate([np.reshape([0] * 20, (-1)), pred], axis=0)
        plot_kwargs = {}
        if 'add_1' in kwargs:
            plot_kwargs['add_1'] = kwargs.get('add_1', False)
        if 'add_0' in kwargs:
            plot_kwargs['add_0'] = kwargs.get('add_0', False)
        disp = CalibrationDisplay.from_predictions(gt, pred,
                                                   n_bins=kwargs.get('n_bins', 5),
                                                   ax=ax_calibration_curve, name=model_name,
                                                   strategy='quantile', **plot_kwargs)


def get_bst_split(X_data: pd.DataFrame, y_data: pd.DataFrame,
                  models: dict, test_size=0.2, metric_fn=accuracy_score, n_trails=10,
                  cv: bool = False, shuffle: bool = False, metric_cut_off: float = None, random_state=None,
                  use_smote: bool = False, **kwargs):
    """
    寻找数据集中最好的数据划分。
    Args:
        X_data: 训练数据
        y_data: 监督数据
        models: 模型名称，Dict类型、
        test_size: 测试集比例
        metric_fn: 评价模型好坏的函数，默认准确率，可选roc_auc_score。
        n_trails: 尝试多少次寻找最佳数据集划分。
        cv: 是否是交叉验证，默认是False，当为True时，n_trails为交叉验证的n_fold
        shuffle: 是否进行随机打乱
        metric_cut_off: 当metric_fn的值达到多少时进行截断。
        random_state: 随机种子
        use_smote: bool, 是否使用SMOTE技术，进行重采样。
        kwargs: 其他模型训练的参数。

    Returns: {'max_idx': max_idx, "max_model": max_model, "max_metric": max_metric, "results": results}

    """
    assert metric_fn in (roc_auc_score, accuracy_score, mean_squared_error)
    if cv and test_size is not None:
        logger.warning('当cv=True的时候，采用的是交叉验证的模式，此时test_size的参数是不生效的，我们将忽略这个test_size设置。'
                       '如果需要手动指定测试集比例，请修改cv=False。')
    results = []
    max_model = None
    max_model_name = None
    max_idx = 0
    max_metric = None
    metrics = {}
    dataset = []
    if not isinstance(X_data, pd.DataFrame) or not isinstance(y_data, pd.DataFrame):
        X_data = pd.DataFrame(X_data)
        y_data = pd.DataFrame(y_data)
        logger.warning('你的数据不是DataFrame类型，可能遇到未知错误！')
    if cv:
        skf = StratifiedKFold(n_splits=n_trails, shuffle=shuffle or random_state is not None, random_state=random_state)
        for train_index, test_index in skf.split(X_data, y_data):
            X_train, X_test = X_data.loc[train_index], X_data.loc[test_index]
            y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]
            dataset.append([X_train, X_test, y_train, y_test])
    for idx in range(n_trails):
        trail = []
        if cv:
            X_train, X_test, y_train, y_test = dataset[idx]
        else:
            rs = None if random_state is None else (idx + random_state)
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=rs)
        X_train_smote, y_train_smote = X_train, y_train
        if use_smote:
            X_train_smote, y_train_smote = smote_resample(X_train, y_train)
        copy_models = copy.deepcopy(models)
        for model_name, model in copy_models.items():
            # model.fit(X_train, y_train)
            # sample_weight = [1 if i == 0 else 0.5 for i in list(np.array(y_train))]
            if kwargs:
                try:
                    model.fit(X_train_smote, y_train_smote, **kwargs)
                    logger.info(f'正在训练{model_name}, 使用{kwargs}。')
                except Exception as e:
                    model.fit(X_train_smote, y_train_smote)
                    logger.warning(f'因为：{e}，训练{model_name}使用{kwargs}失败。')
            else:
                model.fit(X_train_smote, y_train_smote)
            y_pred = model.predict(X_test)
            if metric_fn == roc_auc_score:
                y_proba = model.predict_proba(X_test)[:, 1]
                metric = metric_fn(y_test, y_proba)
            else:
                metric = metric_fn(y_test, y_pred)
            if model_name not in metrics:
                metrics[model_name] = []
            metrics[model_name].append((idx, metric))
            if max_metric is None or metric > max_metric:
                max_metric = metric
                max_idx = idx
                max_model = model
                max_model_name = model_name
            trail.append(metric)
        results.append((trail, (X_train, X_test, y_train, y_test)))
        # 当满足用户需求的时候，可以停止。
        if metric_cut_off is not None and max_metric is not None and max_metric > metric_cut_off:
            logger.info(f'Get best split cut off on {idx + 1} trails!')
            break
    return {'max_idx': max_idx, "max_model": max_model, "max_metric": max_metric, 'max_model_name': max_model_name,
            "results": results, 'metrics': metrics}


def bootstrap_roc(X_data: pd.DataFrame, y_data: pd.DataFrame,
                  models: dict, test_size=0.2, n_trails=10, random_state=None, fpr_resolution: int = 20,
                  alpha: float = 0.95, use_smote: bool = False, **kwargs):
    """
    寻找数据集中最好的数据划分。
    Args:
        X_data: 训练数据
        y_data: 监督数据
        models: 模型名称，Dict类型、
        test_size: 测试集比例
        n_trails: 尝试多少次寻找最佳数据集划分。
        alpha: alpha CI
        fpr_resolution: ROC横坐标的分辨率。
        random_state: 随机种子
        use_smote: bool, 是否使用SMOTE技术，进行重采样。
        kwargs: 其他模型训练的参数。

    Returns: {'max_idx': max_idx, "max_model": max_model, "max_metric": max_metric, "results": results}

    """
    precision = get_param_in_cwd('display.precision', 3)
    metric_fn = roc_auc_score
    ds_split = []
    metrics = defaultdict(list)
    all_fpr = np.linspace(0.0, 1, fpr_resolution)
    if not isinstance(X_data, pd.DataFrame) or not isinstance(y_data, pd.DataFrame):
        X_data = pd.DataFrame(X_data)
        y_data = pd.DataFrame(y_data)
        logger.warning('你的数据不是DataFrame类型，可能遇到未知错误！')
    error_ds = set()
    for idx in range(n_trails):
        rs = None if random_state is None else (idx + random_state)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=rs)
        ds_split.append([X_train, X_test, y_train, y_test])
        X_train_smote, y_train_smote = X_train, y_train
        if use_smote:
            X_train_smote, y_train_smote = smote_resample(X_train, y_train)
        copy_models = copy.deepcopy(models)
        for model_name, model in copy_models.items():
            # model.fit(X_train, y_train)
            # sample_weight = [1 if i == 0 else 0.5 for i in list(np.array(y_train))]
            if kwargs:
                try:
                    model.fit(X_train_smote, y_train_smote, **kwargs)
                    logger.info(f'正在训练{model_name}, 使用{kwargs}。')
                except Exception as e:
                    model.fit(X_train_smote, y_train_smote)
                    logger.warning(f'因为：{e}，训练{model_name}使用{kwargs}失败。')
            else:
                model.fit(X_train_smote, y_train_smote)
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metric = metric_fn(y_test, y_proba)
                fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
                tpr = np.interp(all_fpr, fpr, tpr)
                roc_points = pd.DataFrame(zip(all_fpr, tpr), columns=['1-Specificity', 'Sensitivity'])
                roc_points['model_name'] = model_name
                metrics[model_name].append([idx, y_proba, metric, roc_points])
            except:
                logger.error(f"{model_name} 在RandomState={idx + random_state}.")
                error_ds.add(idx)
    ds_split = [ds for idx, ds in enumerate(ds_split) if idx not in error_ds]
    median = {}
    for model_name in metrics:
        _, _, auc_, points_ = zip(*metrics[model_name])
        auc_median = np.median(auc_[:len(auc_) if len(auc_) % 2 else len(auc_) - 1])
        median[model_name] = auc_.index(auc_median)
        points_ = pd.concat(points_, axis=0).reset_index()
        auc_std = np.std(auc_)
        auc_mean = np.sum(auc_) / len(auc_)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        ci = scipy.stats.norm.ppf(lower_upper_q, loc=auc_mean, scale=auc_std)
        ci[ci > 1] = 1
        ci[ci < 0] = 0
        label_info = f"{model_name} AUC: {{roc_auc:.{precision}f}} ± {{auc_std:.{precision}f}}"
        sns.lineplot(x=points_['1-Specificity'], y=points_['Sensitivity'],
                     label=label_info.format(roc_auc=auc_median, auc_std=auc_std))
    plt.legend(loc="lower right")
    return {'results': ds_split, 'metrics': metrics, 'median_idx': median}


def get_cv_metric_binary_task(models, data_splits, labels, model_names=None):
    """
    生成CV结果，只对二分类模型有效。
    Args:
        models: 需要使用的模型
        data_splits: 数据划分
        labels: 数据对应任务的label
        model_names: 模型名称

    Returns: 所有交叉验证的结果。
    """
    metric = []
    for cv_index, (X_train_sel, X_test_sel, y_train_sel, y_test_sel) in enumerate(data_splits):
        predictions = [[(model.predict(X_train_sel), model.predict(X_test_sel))
                        for model in target] for label, target in zip(labels, models)]
        pred_scores = [[(model.predict_proba(X_train_sel), model.predict_proba(X_test_sel))
                        for model in target] for label, target in zip(labels, models)]

        pred_sel_idx = []
        for model, label, prediction, scores in zip(models, labels, predictions, pred_scores):
            pred_sel_idx_label = []
            if model_names is None:
                model_names = [str(m.__class__) for m in model]
            assert len(prediction) == len(model_names), "模型名称必须与模型长度相同"
            for mname, (train_pred, test_pred), (train_score, test_score) in zip(model_names, prediction, scores):
                # 计算训练集指数
                acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(
                    y_train_sel[label],
                    train_score[:, 1])
                ci = f"{ci[0]:.4f} - {ci[1]:.4f}"
                metric.append((mname, acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres,
                               f"CV-{cv_index}-{label}-train"))

                # 计算验证集指标
                acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(y_test_sel[label],
                                                                                                      test_score[:, 1])
                ci = f"{ci[0]:.4f} - {ci[1]:.4f}"
                metric.append((mname, acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres,
                               f"CV-{cv_index}-{label}-test"))
                # 计算thres对应的sel idx
                pred_sel_idx_label.append(np.logical_or(test_score[:, 0] >= thres, test_score[:, 1] >= thres))

            pred_sel_idx.append(pred_sel_idx_label)
    metric = pd.DataFrame(metric, index=None, columns=['model_name', 'Accuracy', 'AUC', '95% CI',
                                                       'Sensitivity', 'Specificity',
                                                       'PPV', 'NPV', 'Precision', 'Recall', 'F1',
                                                       'Threshold', 'Task'])
    return metric


def create_clf_model(model_names):
    models = {}
    # 判断是纯字符串，使用默认参数进行配置
    if isinstance(model_names, (list, tuple)):
        if 'lr' in model_names or 'LR' in model_names:
            models['LR'] = LogisticRegression(random_state=0)
        # NB
        if 'nb' in model_names or 'NaiveBayes' in model_names:
            models['NaiveBayes'] = GaussianNB()
        # SVM
        if 'svm' in model_names or 'SVM' in model_names:
            models['SVM'] = SVC(probability=True, random_state=0)
        # KNN
        if 'knn' in model_names or 'KNN' in model_names:
            models['KNN'] = KNeighborsClassifier(algorithm='kd_tree')
        # DecisionTree
        if 'dt' in model_names or 'DecisionTree' in model_names:
            models['DecisionTree'] = DecisionTreeClassifier(max_depth=None,
                                                            min_samples_split=2, random_state=0)
        # RandomForest
        if 'rf' in model_names or 'RandomForest' in model_names:
            models['RandomForest'] = RandomForestClassifier(n_estimators=10, max_depth=None,
                                                            min_samples_split=2, random_state=0)
        # ExtraTree
        if 'et' in model_names or 'ExtraTrees' in model_names:
            models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                                                        min_samples_split=2, random_state=0)
        # XGBoost
        if 'xgb' in model_names or 'XGBoost' in model_names:
            models['XGBoost'] = XGBClassifier(n_estimators=10, objective='binary:logistic',
                                              use_label_encoder=False, eval_metric='error')
        # LightGBM
        if 'lgb' in model_names or 'LightGBM' in model_names:
            models['LightGBM'] = LGBMClassifier(n_estimators=10, max_depth=-1, objective='binary')

        # GBM
        if 'gbm' in model_names or 'GradientBoosting' in model_names:
            models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=10, random_state=0)
        # AdaBoost
        if 'adaboost' in model_names or 'AdaBoost' in model_names:
            models['AdaBoost'] = AdaBoostClassifier(n_estimators=10, random_state=0)

        # Multi layer perception
        if 'mlp' in model_names or 'MLP' in model_names:
            models['MLP'] = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, solver='sgd', random_state=0)
    elif isinstance(model_names, dict):
        for model_name, params in model_names.items():
            if 'svm' == model_name or 'SVM' == model_name:
                models['SVM'] = SVC(**params)
            # KNN
            if 'knn' == model_name or 'KNN' == model_name:
                models['KNN'] = KNeighborsClassifier(**params)
            # NB
            if 'nb' == model_name or 'NaiveBayes' == model_name:
                models['NaiveBayes'] = GaussianNB(**params)
            # DecisionTree
            if 'dt' == model_name or 'DecisionTree' == model_name:
                models['DecisionTree'] = DecisionTreeClassifier(**params)
            # RandomForest
            if 'rf' == model_name or 'RandomForest' == model_name:
                models['RandomForest'] = RandomForestClassifier(**params)
            # ExtraTree
            if 'et' == model_name or 'ExtraTrees' == model_name:
                models['ExtraTrees'] = ExtraTreesClassifier(**params)
            # XGBoost
            if 'xgb' == model_name or 'XGBoost' == model_name:
                models['XGBoost'] = XGBClassifier(**params)
            # LightGBM
            if 'lgb' == model_name or 'LightGBM' == model_name:
                models['LightGBM'] = LGBMClassifier(**params)
            # GBM
            if 'gbm' == model_name or 'GradientBoosting' == model_name:
                models['GradientBoosting'] = GradientBoostingClassifier(**params)

            # AdaBoost
            if 'adaboost' == model_name or 'AdaBoost' == model_name:
                models['AdaBoost'] = AdaBoostClassifier(**params)

            # Multi layer perception
            if 'mlp' == model_name or 'MLP' == model_name:
                models['MLP'] = MLPClassifier(**params)
    return models


def create_clf_model_none_overfit(model_names):
    models = {}
    # 判断是纯字符串，使用默认参数进行配置
    if isinstance(model_names, (list, tuple)):
        if 'lr' in model_names or 'LR' in model_names:
            models['LR'] = LogisticRegression(random_state=0)
        # NB
        if 'nb' in model_names or 'NaiveBayes' in model_names:
            models['NaiveBayes'] = GaussianNB()
        # SVM
        if 'svm' in model_names or 'SVM' in model_names:
            models['SVM'] = SVC(probability=True, random_state=0)
        # KNN
        if 'knn' in model_names or 'KNN' in model_names:
            models['KNN'] = KNeighborsClassifier(algorithm='kd_tree')
        # DecisionTree
        if 'dt' in model_names or 'DecisionTree' in model_names:
            models['DecisionTree'] = DecisionTreeClassifier(max_depth=3,
                                                            min_samples_split=2, random_state=0)
        # RandomForest
        if 'rf' in model_names or 'RandomForest' in model_names:
            models['RandomForest'] = RandomForestClassifier(n_estimators=10, max_depth=3,
                                                            min_samples_split=2, random_state=0)
        # ExtraTree
        if 'et' in model_names or 'ExtraTrees' in model_names:
            models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=10, max_depth=3,
                                                        min_samples_split=2, random_state=0)
        # XGBoost
        if 'xgb' in model_names or 'XGBoost' in model_names:
            models['XGBoost'] = XGBClassifier(n_estimators=10, objective='binary:logistic', max_depth=3,
                                              use_label_encoder=False, eval_metric='error')
        # LightGBM
        if 'lgb' in model_names or 'LightGBM' in model_names:
            models['LightGBM'] = LGBMClassifier(n_estimators=10, max_depth=3, objective='binary')

        # GBM
        if 'gbm' in model_names or 'GradientBoosting' in model_names:
            models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=10, random_state=0, max_depth=3)
        # AdaBoost
        if 'adaboost' in model_names or 'AdaBoost' in model_names:
            models['AdaBoost'] = AdaBoostClassifier(n_estimators=10, random_state=0)

        # Multi layer perception
        if 'mlp' in model_names or 'MLP' in model_names:
            models['MLP'] = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, solver='sgd', random_state=0)
    elif isinstance(model_names, dict):
        for model_name, params in model_names.items():
            if 'svm' == model_name or 'SVM' == model_name:
                models['SVM'] = SVC(**params)
            # KNN
            if 'knn' == model_name or 'KNN' == model_name:
                models['KNN'] = KNeighborsClassifier(**params)
            # NB
            if 'nb' == model_name or 'NaiveBayes' == model_name:
                models['NaiveBayes'] = GaussianNB(**params)
            # DecisionTree
            if 'dt' == model_name or 'DecisionTree' == model_name:
                models['DecisionTree'] = DecisionTreeClassifier(**params)
            # RandomForest
            if 'rf' == model_name or 'RandomForest' == model_name:
                models['RandomForest'] = RandomForestClassifier(**params)
            # ExtraTree
            if 'et' == model_name or 'ExtraTrees' == model_name:
                models['ExtraTrees'] = ExtraTreesClassifier(**params)
            # XGBoost
            if 'xgb' == model_name or 'XGBoost' == model_name:
                models['XGBoost'] = XGBClassifier(**params)
            # LightGBM
            if 'lgb' == model_name or 'LightGBM' == model_name:
                models['LightGBM'] = LGBMClassifier(**params)
            # GBM
            if 'gbm' == model_name or 'GradientBoosting' == model_name:
                models['GradientBoosting'] = GradientBoostingClassifier(**params)

            # AdaBoost
            if 'adaboost' == model_name or 'AdaBoost' == model_name:
                models['AdaBoost'] = AdaBoostClassifier(**params)

            # Multi layer perception
            if 'mlp' == model_name or 'MLP' == model_name:
                models['MLP'] = MLPClassifier(**params)
    return models


def create_reg_model(model_names):
    models = {}
    # 判断是纯字符串，使用默认参数进行配置
    if isinstance(model_names, (list, tuple)):
        if 'svm' in model_names or 'SVM' in model_names:
            models['SVM'] = SVR()
        # KNN
        if 'knn' in model_names or 'KNN' in model_names:
            models['KNN'] = KNeighborsRegressor(algorithm='kd_tree')
        # DecisionTree
        if 'dt' in model_names or 'DecisionTree' in model_names:
            models['DecisionTree'] = DecisionTreeRegressor(max_depth=None,
                                                           min_samples_split=2, random_state=0)
        # RandomForest
        if 'rf' in model_names or 'RandomForest' in model_names:
            models['RandomForest'] = RandomForestRegressor(n_estimators=10, max_depth=None,
                                                           min_samples_split=2, random_state=0)
        # ExtraTree
        if 'et' in model_names or 'ExtraTrees' in model_names:
            models['ExtraTrees'] = ExtraTreesRegressor(n_estimators=10, max_depth=None,
                                                       min_samples_split=2, random_state=0)
        # XGBoost
        if 'xgb' in model_names or 'XGBoost' in model_names:
            models['XGBoost'] = XGBRegressor(n_estimators=10, max_depth=5, objective='reg:squarederror',
                                             eval_metric='rmse')
        # LightGBM
        if 'lgb' in model_names or 'LightGBM' in model_names:
            models['LightGBM'] = LGBMRegressor(n_estimators=10, max_depth=4, objective='regression')

        # GBM
        if 'gbm' in model_names or 'GradientBoosting' in model_names:
            models['GradientBoosting'] = GradientBoostingRegressor(n_estimators=10, random_state=0)

        # AdaBoost
        if 'adaboost' in model_names or 'AdaBoost' in model_names:
            models['AdaBoost'] = AdaBoostRegressor(n_estimators=10, random_state=0)

        # Multi layer perception
        if 'mlp' in model_names or 'MLP' in model_names:
            models['MLP'] = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300, solver='sgd')
    elif isinstance(model_names, dict):
        for model_name, params in model_names.items():
            if 'svm' == model_name or 'SVM' == model_name:
                models['SVM'] = SVR(**params)
            # KNN
            if 'knn' == model_name or 'KNN' == model_name:
                models['KNN'] = KNeighborsRegressor(**params)
            # DecisionTree
            if 'dt' == model_name or 'DecisionTree' == model_name:
                models['DecisionTree'] = DecisionTreeRegressor(**params)
            # RandomForest
            if 'rf' == model_name or 'RandomForest' == model_name:
                models['RandomForest'] = RandomForestRegressor(**params)
            # ExtraTree
            if 'et' == model_name or 'ExtraTrees' == model_name:
                models['ExtraTrees'] = ExtraTreesRegressor(**params)
            # XGBoost
            if 'xgb' == model_name or 'XGBoost' == model_name:
                models['XGBoost'] = XGBRegressor(**params)
            # LightGBM
            if 'lgb' == model_name or 'LightGBM' == model_name:
                models['LightGBM'] = LGBMRegressor(**params)
            # GBM
            if 'gbm' == model_name or 'GradientBoosting' == model_name:
                models['GradientBoosting'] = GradientBoostingRegressor(**params)
            # AdaBoost
            if 'adaboost' == model_name or 'AdaBoost' in model_name:
                models['AdaBoost'] = AdaBoostRegressor(**params)
            # Multi layer perception
            if 'mlp' == model_name or 'MLP' == model_name:
                models['MLP'] = MLPRegressor(**params)
    return models


def calc_confusion_matrix(prediction: List[int], gt: List[int], sel_idx: Union[List[bool], np.ndarray] = None,
                          class_mapping: Union[str, dict, list, tuple] = None, num_classes: int = None):
    """

    Args:
        prediction: Prediction of each results.
        gt: Ground truth of each results.
        sel_idx: Use which index of data to calculate cm. default None for all.
        class_mapping: mapping class index to readable classes.
        num_classes: Number of classes.

    Returns:

    """
    num_classes = num_classes or len(set(gt))
    if num_classes != len(set(gt)):
        logger.warning(f'num_classes({num_classes}) is not equal to labels in gt({len(set(gt))}).')
    cm = np.zeros((num_classes, num_classes))
    if sel_idx is not None:
        logger.info(f"使用筛选阈值的数据绘制混淆矩阵！样本量从{len(gt)}变到{sum(sel_idx)}.")
        prediction = prediction[sel_idx]
        gt = gt[sel_idx]
    for pred, y in zip(prediction, gt):
        # cm[int(pred), int(y)] += 1
        cm[int(y), int(pred)] += 1

    mapping = {}
    if isinstance(class_mapping, dict):
        mapping = class_mapping
    elif isinstance(class_mapping, (list, tuple)):
        mapping = dict(enumerate(class_mapping))
    elif class_mapping and os.path.exists(class_mapping):
        label_names = [l_.strip() for l_ in open(class_mapping).readlines()]
        mapping = {i: l for i, l in enumerate(label_names)}
    labels = [mapping[i] if i in mapping else f"label_{i}" for i in range(num_classes)]
    return pd.DataFrame(cm, index=labels, columns=labels)


def draw_matrix(data: pd.DataFrame, norm: bool = False, **kwargs):
    if norm:
        data = data.div(np.sum(data, axis=1), axis=0)
    if 'fmt' not in kwargs:
        kwargs['fmt'] = ".2g" if norm else ".3f"
    if 'col_map' in kwargs:
        col_map = kwargs.pop('col_map')
        data.columns = [col_map[c] if c in col_map else c for c in data.columns]
        data.index = [col_map[c] if c in col_map else c for c in data.index]
    sns.heatmap(data, **kwargs)


def draw_predict_score(pred_score, y_test, threshold=0.5):
    threshold = float(threshold)
    if np.array(pred_score).shape[1] == 1:
        ps = pred_score
    else:
        ps = pred_score[:, 1]
    d = pd.concat([pd.DataFrame(ps), y_test.reset_index()], axis=1)
    # plt.axis('off')
    d.columns = ['predict_score', 'index', 'label']
    d['predict_score'] = (d['predict_score'] - threshold)
    d['predict_score'] = ((d['predict_score'] - d['predict_score'].min()) / (
            d['predict_score'].max() - d['predict_score'].min()) - 0.5) * 2
    d = d.sort_values('predict_score')
    d['index'] = range(d.shape[0])
    plt.bar(d[d['label'] == 0]['index'], d[d['label'] == 0]["predict_score"])
    plt.bar(d[d['label'] == 1]['index'], d[d['label'] == 1]["predict_score"])


def draw_cv_box(data, x, y, hue=None, **kwargs):
    if hue:
        sns.boxplot(data=data, x=x, y=y, hue=hue, **kwargs)
    else:
        sns.boxplot(data=data, x=x, y=y, **kwargs)


def analysis_features(rad_features, labels, not_use=None, legend=None,
                      methods: Union[List[str], str] = 't-SNE', n_neighbors=30, save_dir=None, prefix=''):
    """

    Args:
        rad_features: 待分析的特征
        labels: 每个样本对饮的标签
        not_use: 那些列不使用
        methods: 选择可视化的方法, Random projection, Truncated SVD, Isomap, Standard LLE, Modified LLE,
                MDS, Random Trees, Spectral, t-SNE, NCA
        n_neighbors:
        save_dir: 保存目录
        prefix: 前缀

    Returns:

    """
    if not_use is None:
        not_use = ['ID', 'label']
    if not isinstance(not_use, (list, tuple)):
        not_use = [not_use]
    if methods is not None and not isinstance(methods, (list, tuple)):
        methods = [methods]
    data = np.array(rad_features[[c for c in rad_features.columns if c not in not_use]])
    embeddings = {
        "Random projection": SparseRandomProjection(n_components=2, random_state=42),
        "Truncated SVD": TruncatedSVD(n_components=2),
        "Isomap": manifold.Isomap(n_neighbors=n_neighbors, n_components=2),
        "Standard LLE": manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method="standard"),
        "Modified LLE": manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method="modified"),
        # "Hessian LLE": manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method="hessian"),
        # "LTSA LLE": manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method="ltsa"),
        "MDS": manifold.MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2),
        "Random Trees": make_pipeline(RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
                                      TruncatedSVD(n_components=2), ),
        "Spectral": manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack"),
        "t-SNE": manifold.TSNE(n_components=2, init="pca", learning_rate="auto", n_iter=500,
                               n_iter_without_progress=150, n_jobs=2, random_state=0, ),
        "NCA": NeighborhoodComponentsAnalysis(n_components=2, init="pca", random_state=0),
    }
    projections = {}
    for name, transformer in embeddings.items():
        if methods is None or name in methods:
            try:
                projections[name] = transformer.fit_transform(data, labels)
            except Exception as e:
                logger.error(f"使用{name}进行降维可视化过程中出错。{e}")
    for name in projections:
        X = MinMaxScaler().fit_transform(projections[name])
        color_list = sns.color_palette("hls", len(np.unique(labels)))
        for idx, label in enumerate(np.unique(labels)):
            plt.scatter(*X[labels == label].T, s=60, alpha=0.425, zorder=2, color=color_list[idx])
        plt.legend(labels=legend or np.unique(labels), loc="lower right")
        plt.title(f'Method: {name}')
        if save_dir is not None:
            try:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'{prefix}feature_{name}_viz.svg'), bbox_inches='tight')
            except Exception as e:
                logger.error(f'保存{name}可视化功能失效，因为：「{e}」')
        plt.show()


def select_feature(corr, threshold: float = 0.9, keep: int = 1, verbose: bool = False, topn: int = 1):
    feature_names = corr.columns
    drop_feature_names = [x for x, y in np.array(corr.isna().all().reset_index()) if y]
    has_corr_features = True
    while has_corr_features:
        has_corr_features = False
        corr_fname = {}
        feature2drop = [fname for fname in feature_names if fname not in drop_feature_names]
        for i, fi in enumerate(feature2drop):
            corr_num = 0
            for j in range(i + 1, len(feature2drop)):
                if abs(corr[fi][feature2drop[j]]) > threshold:
                    corr_num += 1
            corr_fname[fi] = corr_num
        corr_fname = sorted(corr_fname.items(), key=lambda x: x[1], reverse=True)
        for fname, corr_num in corr_fname[:topn]:
            if corr_num >= keep:
                has_corr_features = True
                drop_feature_names.append(fname)
                if verbose:
                    logger.info(f'len {len(feature2drop)}, {fname} has {corr_num} features')

    return [fname for fname in feature_names if fname not in drop_feature_names]


def select_feature_mrmr(data: pd.DataFrame, method='MID', num_features: Union[int, float] = 0.1,
                        label_column: str = 'label'):
    """

    Args:
        data: 数据，DataFrame
        method: mrmr的方法，支持MID、MIQ
        num_features: 筛选的特征数量，如果是小数则为百分比，如果是整数则为个数
        label_column: 目标列名，默认为label

    Returns:

    """
    import pymrmr
    if isinstance(num_features, float):
        num_features = max(1, int(data.shape[1] * num_features))
    ordered_column_name = [label_column] + [c for c in data.columns if c != label_column and data[c].dtype != object]
    return pymrmr.mRMR(data[ordered_column_name], method, num_features)


def select_feature_ttest(data: pd.DataFrame, popmean: Union[float, np.ndarray], threshold: float = 0.05):
    feature_names = data.columns
    if not isinstance(popmean, (list, tuple)):
        popmean = [popmean] * len(feature_names)
    elif len(popmean) != len(feature_names):
        raise ValueError('mean is not equal to feature length!')
    res = ttest_1samp(data, popmean)
    return [fname for fname, flag in zip(feature_names, res.pvalue < threshold) if flag]


def lasso_cv_coefs(X_data, y_data, alpha_logmin=-3, points=50, column_names: List[str] = None, cv: int = 10,
                   ensure_lastn: int = None, model_name: str = 'lasso', force_alpha=None,
                   random_state=0, **kwargs):
    """

    Args:
        X_data: 训练数据
        y_data: 监督数据
        alpha_logmin: alpha的log最小值
        points: 打印多少个点。默认50
        column_names: 列名，默认为None，当选择的数据很多的时候，建议不要添加此参数
        cv: 交叉验证次数。
        ensure_lastn: bool, 确保是最后一个MSE不降的alpha
        model_name: 可选使用Lasso还是ElasticNet。
        force_alpha: deprecated!
        random_state: 0
        **kwargs: 其他用于打印控制的参数。

    """
    # 每个特征值随lambda的变化
    alphas = np.logspace(alpha_logmin, 0, points)
    if points != 50 or cv != 10:
        print(f"Points: {points}, CV: {cv}")
    is_multi_task = len(np.array(y_data).shape) > 1 and np.array(y_data).shape[1] > 1
    if model_name.lower() == 'lasso' and is_multi_task:
        lasso_cv = MultiTaskLassoCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=random_state).fit(X_data, y_data)
    elif model_name.lower() == 'lasso' and not is_multi_task:
        lasso_cv = LassoCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=random_state).fit(X_data, y_data)
    elif is_multi_task:
        lasso_cv = MultiTaskElasticNetCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=random_state).fit(X_data, y_data)
    else:
        lasso_cv = ElasticNetCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=random_state).fit(X_data, y_data)
    _, coefs, _ = lasso_cv.path(X_data, y_data, alphas=alphas)
    coefs = np.squeeze(coefs).T
    MSEs = lasso_cv.mse_path_
    MSEs_mean = np.mean(MSEs, axis=1)

    # plt.rcParams['font.sans-serif'] = 'stixgeneral'
    def draw_cv(coefs_):
        plt.semilogx(lasso_cv.alphas_, coefs_, '-', **kwargs)
        if column_names is not None:
            # print(column_names, coefs.shape[1])
            assert len(column_names) == coefs.shape[1]
            plt.legend(labels=column_names, loc='best')
        if ensure_lastn is not None and lasso_cv.alpha_ == 1.0:
            lasso_cv.alpha_ = alphas[-ensure_lastn]
            logger.warning(f'你正在使用ensure_lastn={ensure_lastn}...')
        lambda_info = ''
        if force_alpha is not None:
            lasso_cv.alpha_ = force_alpha
        if lasso_cv.alpha_ != 1.0:
            plt.axvline(lasso_cv.alpha_, color='black', ls="--", **kwargs)
            precision = get_param_in_cwd('display.precision', 4)
            lambda_info = f"(λ={{l:.{precision}f}})"
            lambda_info = lambda_info.format(l=lasso_cv.alpha_)

        plt.xlabel(f'Lambda{lambda_info}')
        plt.ylabel('Coefficients')

    if is_multi_task:
        num_tasks = np.array(y_data).shape[1]
        plt.figure(figsize=(12 * num_tasks, 10))
        for task_idx in range(num_tasks):
            plt.subplot(1, num_tasks, task_idx + 1)
            draw_cv(coefs[:, :, task_idx])
    #         plt.show()
    else:
        draw_cv(coefs)
    return lasso_cv.alpha_


def lasso_cv_efficiency(X_data, y_data, alpha_logmin=-3, points=50, cv: int = 10, ensure_lastn: int = None,
                        model_name: str = 'lasso', force_alpha=None, random_state=0, y_major_locator=0.1, **kwargs):
    """
    Args:
        Xdata: 训练数据
        ydata: 测试数据
        alpha_logmin: alpha的log最小值
        points: 打印的数据密度
        cv: 交叉验证次数
        ensure_lastn: bool, 确保是最后一个MSE不降的alpha
        model_name: 可选使用Lasso还是ElasticNet。
        force_alpha: deprecated!
        random_state: random state
        **kwargs: 其他的图像样式
            # 数据点标记, fmt="o"
            # 数据点大小, ms=3
            # 数据点颜色, mfc="r"
            # 数据点边缘颜色, mec="r"
            # 误差棒颜色, ecolor="b"
            # 误差棒线宽, elinewidth=2
            # 误差棒边界线长度, capsize=2
            # 误差棒边界厚度, capthick=1
    Returns:
    """
    precision = get_param_in_cwd('display.precision', 4)
    alphas = np.logspace(alpha_logmin, 0, points)
    is_multi_task = np.array(y_data).shape[1] > 1
    if model_name.lower() == 'lasso' and is_multi_task:
        lasso_cv = MultiTaskLassoCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=random_state).fit(X_data, y_data)
    elif model_name.lower() == 'lasso' and not is_multi_task:
        lasso_cv = LassoCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=random_state).fit(X_data, y_data)
    elif is_multi_task:
        lasso_cv = MultiTaskElasticNetCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=random_state).fit(X_data, y_data)
    else:
        lasso_cv = ElasticNetCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=random_state).fit(X_data, y_data)
    MSEs = lasso_cv.mse_path_
    MSEs_mean = np.mean(MSEs, axis=1)
    MSEs_std = np.std(MSEs, axis=1)

    # plt.rcParams['figure.figsize'] = (10.0, 8.0)
    default_params = {'fmt': "o", 'ms': 3, 'mfc': 'r', 'mec': 'r', 'ecolor': 'b', 'elinewidth': 2, 'capsize': 2,
                      'capthick': 1}
    default_params.update(kwargs)
    plt.errorbar(lasso_cv.alphas_, MSEs_mean, yerr=MSEs_std, **default_params)
    plt.semilogx()
    if ensure_lastn is not None and lasso_cv.alpha_ == 1.0:
        lasso_cv.alpha_ = alphas[-ensure_lastn]
        logger.warning(f'你正在使用ensure_lastn={ensure_lastn}...')

    lambda_info = ''
    if force_alpha is not None:
        lasso_cv.alpha_ = force_alpha
    if lasso_cv.alpha_ != 1.0:
        plt.axvline(lasso_cv.alpha_, color='black', ls="--", **kwargs)
        lambda_info = f"(λ={{l:.{precision}f}})"
        lambda_info = lambda_info.format(l=lasso_cv.alpha_)
        # lambda_info = f"(λ={:.6f})"

    plt.xlabel(f'Lambda{lambda_info}')
    plt.ylabel('MSE')
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))
    # plt.show()


# def calc_icc(Y, icc_type="icc(2,1)"):
#     """
#     Args:
#         Y: 待计算的数据
#         icc_type: 共支持 icc(2,1), icc(2,k), icc(3,1), icc(3,k)四种
#     """
#
#     [n, k] = Y.shape
#
#     # Degrees of Freedom
#     dfc = k - 1
#     dfe = (n - 1) * (k - 1)
#     dfr = n - 1
#
#     # Sum Square Total
#     mean_Y = np.mean(Y)
#     SST = ((Y - mean_Y) ** 2).sum()
#
#     # create the design matrix for the different levels
#     x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
#     x0 = np.tile(np.eye(n), (k, 1))  # subjects
#     X = np.hstack([x, x0])
#
#     # Sum Square Error
#     predicted_Y = np.dot(
#         np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
#     )
#     print(predicted_Y)
#     residuals = Y.flatten("F") - predicted_Y
#     SSE = (residuals ** 2).sum()
#
#     MSE = SSE / dfe
#
#     # Sum square column effect - between columns
#     SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
#     # MSC = SSC / dfc / n
#     MSC = SSC / dfc
#
#     # Sum Square subject effect - between rows/subjects
#     SSR = SST - SSC - SSE
#     MSR = SSR / dfr
#     if icc_type == "icc(2,1)" or icc_type == 'icc(2,k)':
#         if icc_type == 'icc(2,k)':
#             k = 1
#         ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n + 1e-6)
#     elif icc_type == "icc(3,1)" or icc_type == 'icc(3,k)':
#         if icc_type == 'icc(3,k)':
#             k = 1
#         # print(SSR, dfr, SSE, dfe, MSR, MSE, MSR - MSE)
#         ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + 1e-6)
#     else:
#         raise ValueError(f"icc_type{icc_type} 没有找到！")
#     return max(0, ICC)


def icc_filter_features(dfs: List[DataFrame], threshold: float = 0.9, with_icc: bool = False,
                        columns_start: int = 0, not_icc: List[str] = None):
    """
    ICC校验，取大于threshold阈值的特征
    Args:
        dfs: DataFrame，一般对应于不同人标注的结果。
        threshold: ICC阈值，大于此阈值返回。默认0.9，此值为None，返回每个特征的ICC dict.
        with_icc: Boolean, 是否返回每个特征具体的ICC值。
        columns_start: int，筛选的特征起始列。
        not_icc: 不进行ICC的列名。
    Returns:

    """
    import pingouin as pg
    assert isinstance(dfs, (list, tuple)) and len(dfs) > 1, "做ICC校验的数据至少需要2组数据。"
    assert all(isinstance(df, DataFrame) for df in dfs), "所有的数据必须是DataFrame。"
    assert all(dfs[0].shape == df.shape and len(df.shape) == 2 for df in dfs), '所有的数据维度必须相同'
    columns = dfs[0].columns
    selected_features = {}
    # if isinstance(not_icc, (list, tuple)):
    #     not_icc = [not_icc]
    for idx, column in enumerate(columns):
        if not_icc is not None and column in not_icc:
            logger.info(f'{column}不进行ICC。')
            continue
        if idx >= columns_start:
            try:
                datas = []
                for ridx, df in enumerate(dfs):
                    data_ = df[[column]]
                    data_.insert(0, "reader", np.ones(data_.shape[0]) * (ridx + 1))
                    data_.insert(0, "target", range(data_.shape[0]))
                    datas.append(data_)
                icc_result = pg.intraclass_corr(data=pd.concat(datas),
                                                targets="target", raters="reader", ratings=column)
                selected_features[column] = float(icc_result[icc_result['Type'] == 'ICC1']['ICC'])
            except Exception as e:
                logger.error(f"{column}计算失败，因为：{e}")
    if threshold is not None:
        sel_features = [k for k, v in selected_features.items() if v > threshold]
    else:
        sel_features = selected_features
    if with_icc:
        return sel_features, selected_features
    else:
        return sel_features


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(y_pred_score, y_label, title='Model DCA', labels=None, y_min=None, remap: bool = False, **kwargs):
    """
    plot DCA曲线
    Args:
        y_pred_score: list或者array，为模型预测结果。
        y_label: list或者array，为真实结果。
        title: 图标题
        labels: 图例名称
        y_min: 最小值
        remap: bool
    Returns:

    """
    thresh_group = np.arange(0, 1, 0.01)
    fig, ax = plt.subplots()
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
    net_benefit_model = None
    if not isinstance(y_pred_score, (tuple, list)):
        y_pred_score = [y_pred_score]
    if not isinstance(labels, (tuple, list)):
        labels = [labels]
    if len(y_pred_score) != len(labels):
        logger.warning(f'输入数据量不相等，我们将忽略多余的数据！')

    if remap:
        y_pred_score_ = []
        for y_p_s in y_pred_score:
            if kwargs.get('EX', None) is None:
                model = LogisticRegression()
            else:
                model = RandomForestClassifier(**kwargs.get('EX'))
            if len(y_p_s.shape) == 1:
                y_p_s = np.reshape(np.array(y_p_s), [-1, 1])
                model.fit(y_p_s, y_label)
            else:
                model.fit(y_p_s, y_label)
            scores = model.predict_proba(y_p_s)
            # scores = np.array(normalize_df(pd.DataFrame(scores), method='minmax'))
            y_pred_score_.append(scores[:, 1])
        idx_set = kwargs.get('idx_set', None)
        if isinstance(idx_set, int):
            idx_set = [idx_set]
        y_pred_score = [y if idx_set is None or idx in idx_set else y_pred_score[idx]
                        for idx, y in enumerate(y_pred_score_)]
    for y_pred, label in zip(y_pred_score, labels):
        net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred, y_label)
        ax.plot(thresh_group, net_benefit_model, label=label if label is not None else 'Model')
    ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all')
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')

    if len(y_pred_score) == 1:
        # Fill，显示出模型较于treat all和treat none好的部分
        y2 = np.maximum(net_benefit_all, 0)
        y1 = np.maximum(net_benefit_model, y2)
        ax.fill_between(thresh_group, y1, y2, color='crimson', alpha=0.2)

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0, 1)
    # if y_min is not None:
    #     y_min = max(y_min, net_benefit_model.min() - 0.15)
    if y_min is None:
        y_min = net_benefit_model.min() - 0.15
    ax.set_ylim(y_min, net_benefit_model.max() + 0.15)  # justify the y axis limitation
    ax.set_xlabel(
        xlabel='Threshold Probability',
        fontdict={'family': 'Times New Roman'}
    )
    ax.set_ylabel(
        ylabel='Net Benefit',
        fontdict={'family': 'Times New Roman'}
    )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper right')
    plt.title(title)


def smote_resample(X, y, method='smote'):
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import SMOTE, KMeansSMOTE
    if method == 'smote':
        model = SMOTE(random_state=0)
    elif method == 'smote_enn':
        model = SMOTEENN(random_state=0)
    elif method == 'smote_tomek':
        model = SMOTETomek(random_state=0)
    elif method == 'kmeans_smote':
        model = KMeansSMOTE(random_state=0)
    else:
        raise ValueError('Method not found! 目前仅支持: smote, smote_enn, smote_tomek, kmeans_smote')
    return model.fit_resample(X, y)


def fillna(x: pd.DataFrame, inplace: bool = False, fill_mod: str = 'best', verobse: bool = False):
    """
    填充样本空值
    Args:
        x: 待填充的数据DataFrame
        inplace: 是否直接改变原始x的结果
        fill_mod: 填充模式，目前支持制定数值填充，以及
            * best: 自动寻找最优方式，连续值填充均值，离散值填充中位数。
            * 50%: 中位数填充。
            * mean: 均值填充。
        verobse: 是否输出日志

    Returns:

    """
    assert isinstance(fill_mod, (int, float)) or fill_mod in ['best', '50%', 'mean']
    if inplace:
        data = x
    else:
        data = copy.deepcopy(x)
    desc = data.describe()
    for c in desc.columns:
        if fill_mod == 'best':
            if data[c].dtype == 'float':
                if verobse:
                    logger.info(f"{c}使用【平均值】进行填充")
                fill_value = desc[c]['mean']
            else:
                if verobse:
                    logger.info(f"{c}使用【中位数】进行填充")
                fill_value = desc[c]['50%']
        elif isinstance(fill_mod, (float, int)):
            fill_value = fill_mod
        else:
            fill_value = desc[c][fill_mod]
        data[c] = data[c].fillna(fill_value)
    return data


def plot_feature_importance(model, feature_names=None, save_dir=None, prefix='', trace: bool = False,
                            topk=None, show: bool = True):
    """

    Args:
        model: 模型类
        feature_names: 特征可读名称。
        save_dir: 保存图像目录
        prefix: prefix, 默认为空。
        topk: int， 最重要的k个特征
        trace: 是否跟踪错误信息。
        show: 是否show

    Returns:

    """
    try:
        imp = model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(imp))]
        imp_ = sorted(list(np.array([feature_names, imp]).T), key=lambda x: x[1], reverse=True)[:topk]
        importance = pd.DataFrame(imp_, columns=['feature_name', 'importance'])
        importance['importance'] = importance['importance'].astype(float)
        sns.barplot(y="feature_name", x="importance", data=importance)
        plt.title(model.__class__.__name__)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'{prefix}model_{model.__class__.__name__}_feature_importance.svg'),
                        bbox_inches='tight')
            importance.to_csv(os.path.join(save_dir, f'{prefix}model_{model.__class__.__name__}_importance_weight.csv'),
                              index=False)
        if show:
            plt.show()
        return imp_
    except:
        if trace:
            import traceback
            traceback.print_exc()
        pass


def variable_analysis(data: pd.DataFrame, features: Union[str, List[str]] = None, label_column: str = 'label',
                      need_norm: Union[bool, List[bool]] = False, alpha=0.1, method='uni'):
    """
    进行单因素或者多因素分析，请确保所有的分析特征都是数值类型的。
    Args:
        data: 数据
        features: 需要分析的特征，默认除了ID和label_column列，其他的特征都进行分析。
        label_column: 目标列
        need_norm: 是否标准化所有分析的数据, 默认为False
        alpha: CI alpha, alpha/2 %
        method: 单因素回归分析还是多因素回归分析，uni 单因素回归，multi 多因素回归，默认单因素

    Returns:
    """
    if features is None:
        features = [c for c in data.columns if c not in ['ID', label_column, 'group']]
    if isinstance(features, str):
        features = [features]
    else:
        features = list(features)
    features = [c for c in features if c != label_column]
    assert all(c in data.columns for c in list(features) + [label_column]), \
        f"请检查analysis_features{features}或者label_column({label_column})参数配置，并不是所有的特征都在数据{data.columns}中。"
    if isinstance(need_norm, bool):
        need_norm = [need_norm] * len(features)
    else:
        assert len(need_norm) == len(features), "标准化参数（need_norm）与特征参数（features）长度必须相同！"
    if need_norm:
        not_norm = [label_column] + [feat for nn, feat in zip(need_norm, features) if not nn]
        data = normalize_df(data.copy()[features + [label_column]], not_norm=not_norm)
    else:
        data = data.copy()[features + [label_column]]
    Y = data[label_column]
    features_un = []
    if method == 'uni':
        for c in features:
            model = sm.formula.ols(f'Y~{c}', data=data).fit()
            print_model = model.summary(alpha=alpha)
            column_names = print_model.tables[1].data[0]
            column_names[0] = 'feature_name'
            features_un.append(pd.DataFrame(print_model.tables[1].data[2:], columns=column_names))
    elif method == 'multi':
        model = sm.formula.ols(f'Y~{"+".join(features)}', data=data).fit()
        print_model = model.summary(alpha=alpha)
        column_names = print_model.tables[1].data[0]
        column_names[0] = 'feature_name'
        features_un.append(pd.DataFrame(print_model.tables[1].data[2:], columns=column_names))
    features_un = pd.concat(features_un, axis=0)
    features_un.index = features_un['feature_name']
    for c in features_un.columns[1:]:
        features_un[c] = features_un[c].astype(np.float)
    features_un = features_un.drop(['feature_name', 'std err', 't'], axis=1)
    l_ci = f'lower {int(100 * (1 - alpha / 2))}%CI'
    u_ci = f'upper {int(100 * (1 - alpha / 2))}%CI'
    features_un.columns = ['Log(OR)', 'p_value', l_ci, u_ci]
    features_un['OR'] = features_un['Log(OR)'].map(lambda x: math.exp(x))
    ol_ci = f'OR lower {int(100 * (1 - alpha / 2))}%CI'
    ou_ci = f'OR upper {int(100 * (1 - alpha / 2))}%CI'
    features_un[ol_ci] = features_un[l_ci].map(lambda x: math.exp(x))
    features_un[ou_ci] = features_un[u_ci].map(lambda x: math.exp(x))
    return features_un[['Log(OR)', l_ci, u_ci, 'OR', ol_ci, ou_ci, 'p_value']]


def plot_HR(data, columns=None, hazard_ratios=False, ax=None, coef='OR', alpha=0.1, figsize=None,
            **errorbar_kwargs):
    """
    Produces a visual representation of the coefficients (i.e. log hazard ratios), including
        their standard errors and magnitudes.
    Args:
        data:
        columns: list, optional, specify a subset of the columns to plot
        hazard_ratios: bool, optional. ill present the log-hazard ratios (the coefficients).
            However, by turning this flag to True, the hazard ratios are presented instead.
        ax: matplotlib axis, the matplotlib axis that be edited.
        coef: 系数的列明
        alpha: CI alpha, alpha/2 %
        figsize: 图的尺寸
        **errorbar_kwargs:

    Returns:
        ax: matplotlib axis
            the matplotlib axis that be edited.
    """
    alpha = alpha / 2

    errorbar_kwargs.setdefault("c", "k")
    errorbar_kwargs.setdefault("fmt", "s")
    errorbar_kwargs.setdefault("markerfacecolor", "white")
    errorbar_kwargs.setdefault("markeredgewidth", 1.25)
    errorbar_kwargs.setdefault("elinewidth", 1.25)
    errorbar_kwargs.setdefault("capsize", 3)

    if columns is None:
        columns = data.index
    else:
        data = data.loc[columns]
    data.sort_values(by=coef, inplace=True)
    yaxis_locations = list(range(len(columns)))
    log_hazards = data['Log(OR)'].values.copy()

    if ax is None:
        plt.figure(figsize=figsize or (10, 0.5 * len(columns) + 1))
        ax = plt.gca()

    if hazard_ratios:
        exp_log_hazards = np.exp(log_hazards)
        upper = np.exp(data[f"upper {int(100 * (1 - alpha))}%CI"].values)
        lower = np.exp(data[f"lower {int(100 * (1 - alpha))}%CI"].values)
        ax.errorbar(
            exp_log_hazards,
            yaxis_locations,
            xerr=np.vstack([exp_log_hazards - lower, upper - exp_log_hazards]),
            **errorbar_kwargs,
        )
        ax.set_xlabel("OR (%g%% CI)" % ((1 - alpha) * 100))
    else:
        exp_log_hazards = log_hazards
        xerr = (data[f"upper {int(100 * (1 - alpha))}%CI"].values - data[
            f"lower {int(100 * (1 - alpha))}%CI"].values) / 2
        ax.errorbar(
            exp_log_hazards,
            yaxis_locations,
            xerr=xerr,
            **errorbar_kwargs,
        )
        ax.set_xlabel("log(OR) (%g%% CI)" % ((1 - alpha) * 100))

    best_ylim = ax.get_ylim()
    ax.vlines(1 if hazard_ratios else 0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65, color="k")
    ax.set_ylim(best_ylim)

    tick_labels = data.index

    ax.set_yticks(yaxis_locations)
    ax.set_yticklabels(tick_labels)

    return ax


def uni_multi_variable_analysis(data: pd.DataFrame, features: Union[str, List[str]] = None, label_column: str = 'label',
                                need_norm: Union[bool, List[bool]] = False, alpha=0.1,
                                p_value4multi: float = 0.05, save_dir: Union[str] = None, prefix: str = '',
                                **kwargs):
    """
    单因素，步进多因素分析，使用p_value4multi参数指定多因素分析的阈值
    Args:
        data: 数据
        features: 需要分析的特征，默认除了ID和label_column列，其他的特征都进行分析。
        label_column: 目标列
        need_norm: 是否标准化所有分析的数据, 默认为False
        alpha: CI alpha, alpha/2 %；默认为0.1即95% CI
        p_value4multi: 参数指定多因素分析的阈值，默认为0.05
        save_dir: 保存位置
        prefix: 前缀
        **kwargs:

    Returns:

    """
    if features is None:
        features = [c for c in data.columns if c not in ['ID', label_column, 'group']]
    if isinstance(features, str):
        features = [features]
    for c in features + [label_column]:
        if c not in data.columns:
            raise ValueError(f"请检查analysis_features({c})或者label_column参数配置，并不是所有的特征都在数据中。")

    if isinstance(need_norm, bool):
        need_norm = [need_norm] * len(features)
    else:
        assert len(need_norm) == len(features), "标准化参数（need_norm）与特征参数（features）长度必须相同！"
    uni = variable_analysis(data, features, label_column, need_norm, alpha)
    uni = uni.round(decimals=get_param_in_cwd('display.precision', 3))
    display(uni)
    plot_HR(uni, **kwargs)
    if save_dir is not None:
        save_dir = create_dir_if_not_exists(save_dir)
        plt.savefig(os.path.join(save_dir, f'{prefix}univariable_reg.svg'), bbox_inches='tight')
        uni.to_csv(os.path.join(save_dir, f'{prefix}univariable_reg.csv'))
    plt.show()
    multi_features = list(uni[uni['p_value'] < p_value4multi].index)
    if multi_features:
        need_norm = [nn for nn, feat in zip(need_norm, features) if feat in multi_features]
        multi = variable_analysis(data, multi_features, label_column, need_norm, alpha, method='multi')
        multi = multi.round(decimals=get_param_in_cwd('display.precision', 3))
        display(multi)
        plot_HR(multi, **kwargs)
        if save_dir is not None:
            save_dir = create_dir_if_not_exists(save_dir)
            plt.savefig(os.path.join(save_dir, f'{prefix}multivariable_reg.svg'), bbox_inches='tight')
            multi.to_csv(os.path.join(save_dir, f'{prefix}multivariable_reg.csv'))
        plt.show()
    else:
        logger.warning(f'没有通过单因素分析，找到任何pvalue<{p_value4multi}的特征！')


def get_binary_gt(data, binary_label, binary_col, keep_col=None):
    if keep_col is None:
        keep_col = ['ID']
    if isinstance(data, str) and data.endswith('.csv') and os.path.exists(data):
        data = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        data = copy.deepcopy(data)
    else:
        raise ValueError(f"data参数的数据类型不支持{type(data)}，只支持csv或者DataFrame")
    if not isinstance(keep_col, (list, tuple)):
        keep_col = [keep_col]

    assert all(c in data.columns for c in [binary_col] + list(keep_col)), f"所有的数据必须都在data中！"
    data['label'] = (data[binary_col] == binary_label).astype(np.int)
    return data[list(keep_col) + ['label']]


def get_binary_prediction(data, binary_col, keep_col=None):
    if keep_col is None:
        keep_col = ['ID']
    if isinstance(data, str) and data.endswith('.csv') and os.path.exists(data):
        data = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        data = copy.deepcopy(data)
    else:
        raise ValueError(f"data参数的数据类型不支持{type(data)}，只支持csv或者DataFrame")
    if not isinstance(keep_col, (list, tuple)):
        keep_col = [keep_col]

    assert all(c in data.columns for c in [binary_col] + list(keep_col)), f"所有的数据必须都在data中！"
    data['label-1'] = data[binary_col]
    data['label-0'] = 1 - data['label-1']
    return data[list(keep_col) + ['label-0', 'label-1']]


def merge_results(*results, model_names=None, label_col= None):
    """
    所有的数results必须是ID，label-0，label-1的数据形式
    Args:
        *results:
        model_names:
        label_col: 是否包括label列

    Returns:

    """
    merged_results = results[0]
    for r in results[1:]:
        merged_results = pd.merge(merged_results, r, on='ID', how='inner')
    if model_names is not None:
        assert len(model_names) == len(results)
        column_names = ['ID']
        for idx, model_name in enumerate(model_names):
            if label_col is not None:
                column_names.extend([f"label-0-{idx}", model_name, label_col if idx == 0 else f"{label_col}-{idx}"])
            else:
                column_names.extend([f"label-0-{idx}", model_name])
        merged_results.columns = column_names
    return merged_results


def human_boost(data, x_column, hue, metrics, save_dir='img', ylim=0):
    """

    Args:
        data: 数据
        x_column: X坐标的列名
        hue: 使用什么列名来区分系列
        metrics: 要评估那些指标。所有的指标必须要在data中。
        save_dir: 保存位置，默认位当前目录的img文件夹。
        ylim: Y轴的最小值。

    Returns:

    """
    assert all(m in data.columns for m in metrics), f"所有的metrics:{metrics}，必须都在data中！"
    for m in metrics:
        ax = sns.catplot(
            data=data.reset_index(), kind="point",
            x=x_column, y=m,
            hue=hue
        )
        ax.set_xlabels(visible=False)  # .set_visible(False)
        plt.ylim(ylim)
        ax.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'human_boost_{m}.svg'), bbox_inches='tight')
        plt.show()


init_CN()
