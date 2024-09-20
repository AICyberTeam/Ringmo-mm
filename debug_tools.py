import torch
import numpy as np

g_whole_tensor = False
g_whole_list = False
g_whole_dict = False
g_max_dict_key_num = 10
g_max_list_ele_num = 10
g_max_tensor_ele_num = 50


def _level_show(level, s):
    print(f"{'     ' * level}{s}")


def _show_dict(dic, level=0):
    tail_idx = len(dic.keys()) if g_whole_dict else g_max_dict_key_num
    for key, value in list(dic.items())[:tail_idx]:
        _level_show(level, f"{key}:")
        _show_value(value, level + 1)


def _show_tensor(value, level):
    element_num = 1
    for shp in value.shape:
        element_num *= shp
    _level_show(level, f"{type(value)} of shape {value.shape}, "
                       f"{value if (element_num <= g_max_tensor_ele_num or g_whole_tensor) else ''}")


def _show_array(value, level):
    _level_show(level, f'{type(value)} with length {len(value)}')
    tail_idx = len(value) if g_whole_list else g_max_list_ele_num
    for i, v in enumerate(value[:tail_idx]):
        _level_show(level + 1, f'the No.{i + 1}th element')
        _show_value(v, level + 1)


def _show_value(value, level=0):
    if isinstance(value, list) or isinstance(value, tuple):
        _show_array(value, level)
    elif isinstance(value, dict):
        _show_dict(value, level)
    elif isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
        _show_tensor(value, level)
    else:
        _level_show(level, f'the value is {value}')


def show_value(value, title='',
               whole_tensor=False,
               whole_list=False,
               whole_dict=False,
               max_dict_key_num=10,
               max_list_ele_num=10,
               max_tensor_ele_num=50):
    global g_whole_tensor, g_whole_dict, g_whole_list, g_max_tensor_ele_num, g_max_list_ele_num, g_max_dict_key_num
    g_whole_tensor, g_whole_dict, g_whole_list, g_max_tensor_ele_num, g_max_list_ele_num, g_max_dict_key_num = \
        whole_tensor, whole_dict, whole_list, max_tensor_ele_num, max_list_ele_num, max_dict_key_num
    print('*' * 100)
    if title:
        print(f'the variable "{title}" is shown bottom:')
    _show_value(value)
