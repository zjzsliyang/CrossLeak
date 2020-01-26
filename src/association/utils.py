import os
import re
import sys
import yaml
import pickle
import logging
import warnings
import traceback
import collections


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def get_config(project_dir: str):
    with open(os.path.join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)
    return cfg


def get_parent_folder_name(direction: str, num=2):
    for i in range(num):
        path = os.path.split(direction)
        direction = path[0]
        result = path[1]
    return result


# get files of pattern from piles_num children of root_path
def get_format_subfile(root_path: str, piles_num: int, pattern: str):
    paths = [root_path]
    for i in range(piles_num):
        paths = list(filter(lambda x: not os.path.isfile(x), paths))
        num = 0
        te = paths.copy()
        for path in te:
            temp = os.listdir(path)
            temp = list(map(lambda x: os.path.join(path, x), temp))
            paths.extend(temp)
            num += 1
        for k in range(num):
            paths.pop(0)
    paths = list(filter(lambda x: os.path.isfile(x) and re.match(pattern, x), paths))
    return paths


def get_meeting_and_path(path: str, pattern: str):
    paths = get_format_subfile(path, 2, pattern)
    result = {}
    for path in paths:
        result[get_parent_folder_name(path, 2)] = path
    return result


# TODO: fix ad-hoc function
def get_meeting_poi_name(path: str, real_world: bool, thres=''):
    if real_world:
        # real world dataset store POI name in png
        wifi_paths = get_format_subfile(path, 2, r'.+' + re.escape(str(thres)) + r'.+\.png$')
        result = collections.defaultdict(set)
        for wifi_path in wifi_paths:
            name = get_parent_folder_name(wifi_path, 1)
            meeting = get_parent_folder_name(wifi_path, 2)
            # split by _ or . to get the people name
            for peo in re.split('[_.]', name)[1:-1]:
                if peo is not '':
                    result[meeting].add(peo)
        new_res = {}
        for meeting, peos in result.items():
            new_res[meeting] = list(peos)
        return new_res
    else:
        # fake dataset store POI name in txt
        wifi_paths = get_format_subfile(path, 2, r'.+\.txt')
        result = {}
        for wifi_path in wifi_paths:
            people_name = open(wifi_path, 'r').read()
            meeting = get_parent_folder_name(wifi_path, 2)
            result[meeting] = people_name.split('_')
        return result


def get_meeting_all_people_mac(path: str, real_world: bool, thres=''):
    if real_world:
        # real world dataset store all mac address with thres rssi in csv
        wifi_paths = get_format_subfile(path, 2, r'.+\.csv')
        result = {}
        for wifi_path in wifi_paths:
            if str(thres) + '.csv' in wifi_path:
                people_name = open(wifi_path, 'r').read()
                meeting = get_parent_folder_name(wifi_path, 2)
                if thres == '':
                    result[meeting] = re.split(', ', people_name)
                else:
                    result[meeting] = re.split('[_.]', people_name)
        return result


def get_meeting_poi_num(path: str, real_world: bool, thres=''):
    result = get_meeting_poi_name(path, real_world, thres)
    return {meeting: len(people_name) for meeting, people_name in result.items()}


def get_iou_num(pic: str, real_world: bool):
    if real_world:
        return int(re.split('[._\-]', pic)[-2])
    else:
        return int(re.split('[._\-]', pic)[-3])


def parse(tree, p, all_paths):
    path = p[:]
    path.append(str(tree.get_id()))
    if tree.is_leaf():
        all_paths.append(path)
    else:
        # Here assume get_left() returns some false value for no left child
        left = tree.get_left()
        if left:
            parse(left, path, all_paths)
        right = tree.get_right()
        if right:
            parse(right, path, all_paths)
    return all_paths


def load_true_label(audio: bool, project_dir: str):
    # load audio / video true label
    if audio:
        true_label = {}
        # count for error
        dia_err = 0
        dia_segs = []
        list_pth = os.path.join(project_dir, 'middle_data', 'list.txt')
        gt_dir = os.path.join(project_dir, 'dia_groundtruth')
        with open(list_pth, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                l = line.split(' ')
                dia_segs.append(l[1])
            for seg in dia_segs:
                meeting = seg.split('/')[0]
                timestamp = seg.split('/')[2]
                if 'seg' not in timestamp:
                    start = int(timestamp.split('_')[0])
                    end = int(timestamp.split('_')[1][:-4])
                    gt_pth = os.path.join(gt_dir, '%s.csv' % meeting)
                    gt_timestamps = []
                    gt_speakers = []
                    with open(gt_pth, 'r', encoding='utf-8-sig') as f:
                        true_label[meeting] = {}
                        lines = f.read().splitlines()
                        for inx in range(len(lines) - 1):
                            gt_timestamps.append(lines[inx + 1].split(',')[1])
                            gt_speakers.append(lines[inx].split(',')[2])
                        true_label[meeting]['timestamp']= gt_timestamps
                        true_label[meeting]['speaker'] = gt_speakers
                    for t in gt_timestamps:
                        if len(t) > 0:
                            tick = int(float(t) * 1000)
                            if start < tick:
                                # check if diarization is incorrect (contain more than one speaker)
                                if end > tick:
                                    dia_err += 1
            logging.info('dia error = %d' % dia_err)
    else:
        with open(os.path.join(project_dir, 'middle_data', 'true_label.pk'), 'rb') as f:
            true_label = pickle.load(f)
        true_cnt = {}
        for item, peo in true_label.items():
            true_cnt[peo] = true_cnt.get(peo, 0) + 1
        logging.info('true_cnt: {}'.format(true_cnt))
    return true_label