# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2019/7/22
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2019 All Rights Reserved.
import argparse
import json
import os
import shutil
from functools import partial
from typing import Iterable
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from monai.data import DataLoader
from monai.transforms import Compose, ScaleIntensity, AddChannel, Resize, EnsureType
from sklearn.model_selection import train_test_split, StratifiedKFold

from onekey_algo.datasets.image_loader import default_loader
from onekey_algo.utils.about_log import logger
from onekey_core.core import create_model
from onekey_core.core import create_standard_image_transformer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def extract(samples, model, transformer, device=None, fp=None):
    results = []
    # Inference
    if not isinstance(samples, (list, tuple)):
        samples = [samples]
    with torch.set_grad_enabled(False):
        for sample in samples:
            fp.write(f"{os.path.basename(sample)},")
            sample_ = transformer(default_loader(sample))
            sample_ = sample_.to(device)
            # print(sample_.size())
            outputs = model(sample_.view(1, *sample_.size()))
            results.append(outputs)
    return results


def extract3d(samples: DataLoader, images_files, model, device=None, fp=None):
    results = []
    # Inference
    with torch.set_grad_enabled(False):
        for sample, fnames in zip(samples, images_files):
            fp.write(f"{os.path.basename(fnames)},")
            outputs = model(sample.to(device))
            results.append(outputs)
    return results


def print_feature_hook(module, inp, outp, fp, post_process=None):
    features = outp.cpu().numpy()
    if post_process is not None:
        features = post_process(features)
    print(','.join(map(lambda x: f"{x:.6f}", np.reshape(features, -1))), file=fp)


def reg_hook_on_module(name, model, hook):
    find_ = 0
    for n, m in model.named_modules():
        if name == n:
            m.register_forward_hook(hook)
            find_ += 1
    if find_ == 0:
        logger.warning(f'{name} not found in {model}')
    elif find_ > 1:
        logger.info(f'Found {find_} features named {name} in {model}')
    return find_


def init_from_model3d(model_name, pretrained=None, num_classes=400, img_size=(96, 96, 96), **kwargs):
    # Configuration of core
    kwargs.update({'pretrained': pretrained, 'model_name': model_name, 'num_classes': num_classes})
    model = create_model(**kwargs).eval()
    # Config device automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if isinstance(pretrained, str) and os.path.exists(pretrained):
        state_dict = torch.load(pretrained, map_location=device)['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f'成功加载{pretrained}模型参数。')
    transformer = Compose([ScaleIntensity(), AddChannel(), Resize(img_size), EnsureType()])
    return model, transformer, device


def init_from_model(model_name, model_path=None, num_classes=1000, model_state='model_state_dict',
                    img_size=(224, 224), **kwargs):
    # Configuration of core
    kwargs.update({'pretrained': True if model_path is None else False,
                   'model_name': model_name, 'num_classes': num_classes})
    model = create_model(**kwargs).eval()
    # Config device automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)[model_state]
        model.load_state_dict(state_dict)
    if 'inception' in model_name.lower():
        if isinstance(img_size, int):
            if img_size != 299:
                logger.warning(f'{model_name} is inception structure, `img_size` is set to be 299 * 299.')
                img_size = 299
        elif isinstance(img_size, Iterable):
            if 299 not in img_size:
                logger.warning(f'{model_name} is inception structure, `img_size` is set to be 299 * 299.')
                img_size = (299, 299)
    transformer = create_standard_image_transformer(img_size, phase='valid')
    return model, transformer, device


def init_from_onekey3d(config_path):
    config = json.loads(open(os.path.join(config_path, 'task.json')).read())
    model_path = os.path.join(config_path, 'BEST-training-params.pth')
    assert 'model_name' in config and 'num_classes' in config and 'input_size' in config
    model, transformer, device = init_from_model3d(model_name=f"classification3d.{config['model_name']}",
                                                   pretrained=model_path,
                                                   num_classes=config['num_classes'],
                                                   img_size=config['input_size'],
                                                   in_channels=config['in_channels'])
    return model, transformer, device


def init_from_onekey(config_path):
    config = json.loads(open(os.path.join(config_path, 'task.json')).read())
    model_path = os.path.join(config_path, 'resnet34-attent.pth')
    assert 'model_name' in config and 'num_classes' in config and 'transform' in config
    # Configuration of transformer.
    transform_config = {'phase': 'valid'}
    transform_config.update(config['transform'])
    assert 'input_size' in transform_config, '`input_size` must in `transform`'
    transformer = create_standard_image_transformer(**transform_config)

    # Configuration of core
    model_config = {'pretrained': False, 'model_name': config['model_name'], 'num_classes': config['num_classes']}
    if 'vit' in config['model_name'].lower():
        model_config.update(config['vit_settings'])
    model = create_model(**model_config)
    # Configuration of device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    state_dict = torch.load(model_path, map_location=device)['model_state_dict']
    for key in list(state_dict.keys()):
        if key.startswith('module.'):
            new_key = key[7:]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
    model.eval()
    return model, transformer, device


def split_dataset2folder_by_comp1(label_data: pd.DataFrame, crop_path, train_file, val_file, label='label'):
    """

    Args:
        label_data:
        crop_path:
        train_file:
        val_file:
        label:

    Returns:

    """
    assert label in label_data.columns and 'ID' in label_data.columns
    root = os.path.basename(crop_path)
    train_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')

    train = pd.read_csv(train_file, header=0)
    test = pd.read_csv(val_file, header=0)

    for l in np.unique(label_data[label]):
        os.makedirs(os.path.join(root, rf'train\{l}'), exist_ok=True)
        os.makedirs(os.path.join(root, rf'val\{l}'), exist_ok=True)

    label_data = label_data[['ID', label]]
    mapping = {x: str(y) for x, y in np.array(label_data)}

    for sample in train['ID']:
        if os.path.exists(os.path.join(crop_path, sample.replace('.gz', '.png'))):
            shutil.copy(os.path.join(crop_path, sample.replace('.gz', '.png')), rf'{train_path}\{mapping[sample]}')
    for sample in test['ID']:
        if os.path.exists(os.path.join(crop_path, sample.replace('.gz', '.png'))):
            shutil.copy(os.path.join(crop_path, sample.replace('.gz', '.png')), rf'{val_path}\{mapping[sample]}')


def split_dataset2list_by_comp1(label_data: pd.DataFrame, train_file, val_file, root='./', label='label',
                                id_map=lambda x: x.replace('.gz', '.png')):
    """

    Args:
        label_data:
        train_file:
        val_file:
        root: 存放数据目录
        label:
        id_map: ID的映射函数

    Returns:

    """
    assert label in label_data.columns and 'ID' in label_data.columns
    train_path = os.path.join(root, 'train.txt')
    val_path = os.path.join(root, 'val.txt')

    train = pd.merge(pd.read_csv(train_file, header=0), label_data, on='ID', how='inner')
    train['ID'] = train['ID'].map(id_map)
    val = pd.merge(pd.read_csv(val_file, header=0), label_data, on='ID', how='inner')
    val['ID'] = val['ID'].map(id_map)
    with open(os.path.join(root, 'labels.txt'), 'w') as f:
        for l in sorted(np.unique(label_data[label])):
            print(l, file=f)

    for p, d in ([(train_path, train), (val_path, val)]):
        d[['ID', label]].to_csv(p, header=False, index=False, sep='\t')


def show_cam_on_image(img: Union[np.ndarray, Image.Image],
                      mask: np.ndarray,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_JET,
                      reverse: bool = False,
                      heatmap_only: bool = False) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap. By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param reverse: Reverse jet color
    :param heatmap_only: 是否只要heatmap
    :returns: The default image with the cam overlay.
    """
    try:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    except:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask / (np.max(np.array(mask)) + 1e-6)), colormap)
    if reverse:
        heatmap = 255 - heatmap
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    if isinstance(img, Image.Image):
        img = np.array(img) / 255
    elif isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = np.array(Image.fromarray(255 * img / (np.max(img) + 1e-6)).convert('RGB')) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + (img if not heatmap_only else 0)
    cam = cam / np.max(cam)
    return cam


def split_dataset(X_data: pd.DataFrame, y_data: pd.DataFrame = None, test_size=0.2, n_trails=10,
                  cv: bool = False, shuffle: bool = True, random_state=0, save_dir=None, prefix=''):
    """
    数据划分。
    Args:
        X_data: 训练数据
        y_data: 监督数据
        test_size: 测试集比例
        n_trails: 尝试多少次寻找最佳数据集划分。
        cv: 是否是交叉验证，默认是False，当为True时，n_trails为交叉验证的n_fold
        shuffle: 是否进行随机打乱
        random_state: 随机种子
        prefix: 保存文件名的前缀，默认为空
        save_dir: 信息保存的路径。

    Returns:

    """
    if cv and y_data is None:
        raise ValueError(f'使用交叉验证，必须指定y！')
    dataset = []
    if cv:
        skf = StratifiedKFold(n_splits=n_trails, shuffle=shuffle or random_state is not None, random_state=random_state)
        for train_index, test_index in skf.split(X_data, y_data):
            X_train, X_test = X_data.loc[train_index], X_data.loc[test_index]
            y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]
            dataset.append([X_train, X_test, y_train, y_test])
    trails = []
    if save_dir:
        save_dir = os.path.join(save_dir, 'split_info')
        os.makedirs(save_dir, exist_ok=True)
    for idx in range(n_trails):
        if cv:
            X_train, X_test, y_train, y_test = dataset[idx]
        else:
            rs = None if random_state is None else (idx + random_state)
            X_train, X_test = train_test_split(X_data, test_size=test_size, random_state=rs, shuffle=shuffle)
        trails.append([X_train, X_test])
        if save_dir:
            cv_info = '-CV' if cv else '-RND'
            X_train.to_csv(os.path.join(save_dir, f'{prefix}train{cv_info}-{idx}.txt'),
                           index=False, header=False, sep='\t')
            X_test.to_csv(os.path.join(save_dir, f'{prefix}val{cv_info}-{idx}.txt'),
                          index=False, header=False, sep='\t')
    return trails


def split_dataset4sol(X_data: pd.DataFrame, y_data: pd.DataFrame = None, test_size=0.2, n_trails=10,
                      cv: bool = False, shuffle: bool = False, random_state=0, save_dir=None, prefix='',
                      map_ext: Union[bool, str] = True):
    """
    数据划分。
    Args:
        X_data: 训练数据
        y_data: 监督数据
        test_size: 测试集比例
        n_trails: 尝试多少次寻找最佳数据集划分。
        cv: 是否是交叉验证，默认是False，当为True时，n_trails为交叉验证的n_fold
        shuffle: 是否进行随机打乱
        random_state: 随机种子
        prefix: 保存文件名的前缀，默认为空
        save_dir: 信息保存的路径。
        map_ext: 是否映射扩展名，默认修改成png。

    Returns:

    """
    if cv and y_data is None:
        raise ValueError(f'使用交叉验证，必须指定y！')
    dataset = []
    X_data = X_data.reset_index(drop=True)
    if y_data is not None:
        y_data = y_data.reset_index(drop=True)
    if cv:
        skf = StratifiedKFold(n_splits=n_trails, shuffle=shuffle or random_state is not None, random_state=random_state)
        for train_index, test_index in skf.split(X_data, y_data):
            X_train, X_test = X_data.loc[train_index], X_data.loc[test_index]
            y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]
            dataset.append([X_train, X_test, y_train, y_test])
    trails = []
    if save_dir:
        save_dir = os.path.join(save_dir, 'split_info')
        os.makedirs(save_dir, exist_ok=True)
    for idx in range(n_trails):
        if cv:
            X_train, X_test, y_train, y_test = dataset[idx]
        else:
            rs = None if random_state is None else (idx + random_state)
            X_train, X_test = train_test_split(X_data, test_size=test_size, random_state=rs)
        trails.append([X_train, X_test])
        if save_dir:
            cv_info = '-CV' if cv else '-RND'
            X_train4dl = X_train.copy()
            X_test4dl = X_test.copy()
            if isinstance(map_ext, bool) and map_ext:
                X_train4dl['ID'] = X_train4dl['ID'].map(lambda x: f"{os.path.splitext(str(x))[0]}.png")
                X_test4dl['ID'] = X_test4dl['ID'].map(lambda x: f"{os.path.splitext(str(x))[0]}.png")
            elif isinstance(map_ext, str):
                X_train4dl['ID'] = X_train4dl['ID'].map(lambda x: f"{os.path.splitext(str(x))[0]}{map_ext}")
                X_test4dl['ID'] = X_test4dl['ID'].map(lambda x: f"{os.path.splitext(str(x))[0]}{map_ext}")

            X_train4dl.to_csv(os.path.join(save_dir, f'{prefix}train{cv_info}-{idx}.txt'),
                              index=False, header=False, sep='\t')
            X_test4dl.to_csv(os.path.join(save_dir, f'{prefix}val{cv_info}-{idx}.txt'),
                             index=False, header=False, sep='\t')

            X_train['group'] = 'train'
            X_test['group'] = 'test'
            pd.concat([X_train, X_test], axis=0).to_csv(os.path.join(save_dir, f"{prefix}label{cv_info}-{idx}.csv"),
                                                        index=False, encoding='utf-8-sig')
    return trails


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Inference')

    parser.add_argument('-c', '--config_path', dest='c', default='20211014/resnet18/viz',
                        help='Model and transformer configuration')
    parser.add_argument('-d', '--directory', dest='d',
                        default=r'C:\Users\onekey\Project\data\labelme', help='Inference data directory.')
    parser.add_argument('-l', '--list_file', dest='l', default=None, help='Inference data list file')

    args = parser.parse_args()
    if args.d is not None:
        test_samples = [os.path.join(args.d, p) for p in os.listdir(args.d) if p.endswith('.jpg')]
    elif args.l is not None:
        with open(args.l) as f:
            test_samples = [l.strip() for l in f.readlines()]
    else:
        raise ValueError('You must provide a directory or list file for inference.')
    model_name = 'resnet18'
    model, transformer, device = init_from_model(model_name=model_name)
    # print(model)
    # for n, m in model.named_modules():
    #     print(n, m)
    feature_name = 'avgpool'
    outfile = open('feature.txt', 'w')
    hook = partial(print_feature_hook, fp=outfile)
    find_num = reg_hook_on_module(feature_name, model, hook)
    results_ = extract(test_samples[:5], model, transformer, device, fp=outfile)
    print(json.dumps(results_, ensure_ascii=False, indent=True))
