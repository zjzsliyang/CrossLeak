import os
import utils
import bisect
import pickle
import logging
import operator
import functools
import numpy as np
from . import utils
from pulp import *
from termcolor import colored
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, to_tree


# Simultaneously Clustering And Naming
def scan(bio_features, bio_paths, bio_info, real_world, cdist_metric, victims_thres, omega, meeting_num, meeting_thres, meeting_index, wifi_people_in_meetings, poi_people, project_dir, audio):
    logging.info('begin SCAN with parameters: ')

    linkage_mat = linkage(bio_features, method='average')
    root_node, nodelist = to_tree(linkage_mat, rd=True)

    # decide the starting position of useful IDs to avoid tiny clusters
    if real_world:
        frac_start = 0.995
    else:
        frac_start = 0.95
    start_node_idx = int(len(nodelist) * frac_start)

    # assign bio to useful nodes (clusters), avoiding meaningless association
    stat = {}
    cluster_feature = {}
    useful_nodelist, useful_id = [], 0
    for node in nodelist:
        if node.id < start_node_idx:
            continue
        node.useful_id = useful_id
        node.bios = set()
        stat[useful_id] = set()
        cluster_feature[useful_id] = []
        for leaf in node.pre_order():
            bio_path = bio_paths[leaf]
            cluster_feature[useful_id].append(bio_info[leaf])
            #TODO: may differ in audio and video
            parent = utils.get_parent_folder_name(bio_path[0], 3)
            tem = parent.split('_')
            # node bios: {'04-03-11-00-00_04-03-14-00-00', '03-29-14-00-00_03-29-17-00-00'}
            node.bios.add('%s_%s' % (tem[0], tem[1]))
            stat[useful_id].add('%s_%s' % (tem[0], tem[1]))
        useful_nodelist.append(node)
        useful_id += 1

    # dict to save
    res_stat = {}
    for num in stat:
        res_stat[num] = {}
        for n in stat[num]:
            res_stat[num][n] = []

    bio_people_in_meetings = np.zeros([len(useful_nodelist), min(meeting_num, meeting_thres)])
    for node in useful_nodelist:
        for bio in node.bios:
            if meeting_index[bio] < meeting_thres:
                bio_people_in_meetings[node.useful_id, meeting_index[bio]] = 1
        for bio_idx in node.pre_order():
            bio_path = bio_paths[bio_idx]
            res_stat[node.useful_id][bio].append(bio_path)

    # get the association cost matrix
    logging.info('bio_people_in_meetings shape: {}'.format(bio_people_in_meetings.shape))
    logging.info('wifi_people_in_meetings shape: {}'.format(wifi_people_in_meetings.shape))

    # get the association cost matrix
    dist_mat = cdist(bio_people_in_meetings, wifi_people_in_meetings, metric=cdist_metric)
    logging.info('dist_mat shape: {} with min: {} and max: {}'.format(dist_mat.shape, dist_mat.min(), dist_mat.max()))

    # PuLP Part
    rows = [str(i) for i in range(dist_mat.shape[0])] # stands for bio_feature
    cols = [str(i) for i in range(dist_mat.shape[1])] # stands for poi_people
    all_paths_to_leaves = utils.parse(to_tree(linkage_mat), [], [])
    # init
    prob = pulp.LpProblem("scan_opt", pulp.LpMinimize)
    assignment = pulp.LpVariable.dicts('assignment', (rows, cols), cat="Binary")
    # obj function
    prob += pulp.lpSum([assignment[r][c] * (dist_mat[int(r)][int(c)] + omega * useful_nodelist[int(r)].dist)
                        for r in rows for c in cols])

    # enforce constraints for useful nodes (clusters) and IDs respectively
    for r in rows:
        prob += pulp.lpSum([assignment[r][c] for c in cols]) <= 1
    for c in cols:
        prob += pulp.lpSum([assignment[r][c] for r in rows]) <= 1

    logging.info('using victims_thres: {}'.format(victims_thres))
    prob += pulp.lpSum([assignment[r][c] for r in rows for c in cols]) == victims_thres

    # enforce the hierarchy relation constraints between nodes
    for leaf_path in all_paths_to_leaves:
        prob += pulp.lpSum([assignment[str(int(r) - start_node_idx)][c]
                            for r in leaf_path for c in cols if int(r) >= start_node_idx]) <= 1

    # pulp solve
    prob.solve()
    logging.info(pulp.LpStatus[prob.status])

    res_peo = []
    res_pic = []
    for r in rows:
        for c in cols:
            if assignment[r][c].varValue > 0:
                logging.debug('chose r: {}, c: {}, value: {}'.format(r, c, assignment[r][c].varValue))
                res_peo.append(int(c))
                res_pic.append(int(r))

    # assign results
    nb_labelled_images = 0
    save_matrix = {}
    labeled = set()
    final_res = collections.OrderedDict()
    for r in rows:
        for c in cols:
            if assignment[r][c].varValue > 0:
                if int(c) < len(poi_people):
                    name = poi_people[int(c)]
                else:
                    name = 'non_poi_' + c
                final_res[name] = res_stat[int(r)]
                save_matrix[name] = np.mean(np.array(cluster_feature[int(r)]), axis=0)
                nb_labelled_images += len(useful_nodelist[int(r)].pre_order())
                labeled.update(set(useful_nodelist[int(r)].pre_order()))

    # load true label
    true_label = utils.load_true_label(audio, project_dir)

    peo_in_cluster = defaultdict(dict)
    face_in_cluster = np.zeros([len(final_res), meeting_num])
    face_name = []
    if audio:
        if real_world:
            for peo, item in final_res.items():
                for full_pic in functools.reduce(operator.concat, item.values()):
                    if full_pic.split('/')[-3] in meeting_index:
                        if peo not in face_name:
                            face_name.append(peo)
                        face_in_cluster[face_name.index(peo)][meeting_index[full_pic.split('/')[-3]]] = 1
                    if full_pic.split('/')[-3] in true_label:
                        index1 = bisect.bisect_left(true_label[full_pic.split('/')[-3]]['timestamp'], full_pic.split('/')[-1].split('_')[0])
                        index2 = bisect.bisect_left(true_label[full_pic.split('/')[-3]]['timestamp'], full_pic.split('/')[-1].split('_')[1][:-4])
                        if index1 == index2:
                            if index1 < len(true_label[full_pic.split('/')[-3]]['speaker']):
                                peo_in_cluster[peo][true_label[full_pic.split('/')[-3]]['speaker'][index1]] = peo_in_cluster[peo].get(true_label[full_pic.split('/')[-3]]['speaker'][index1], 0) + 1
        else:
            for peo, item in final_res.items():
                for meeting, pics in item.items():
                    for full_pic in pics:
                        # print('full_pic: {}'.format(full_pic))
                        # print('meeting_index: {}'.format(meeting_index.keys()))
                        if full_pic.split('/')[-3] in meeting_index:
                            if peo not in face_name:
                                face_name.append(peo)
                            face_in_cluster[face_name.index(peo)][meeting_index[full_pic.split('/')[-3]]] = 1
                        id = full_pic.split('/')[-1].split('_')[0]
                        peo_in_cluster[peo][id] = peo_in_cluster[peo].get(id, 0) + 1
                        # if full_pic.split('/')[-1] in true_label:
                        #     peo_in_cluster[peo][true_label[full_pic.split('/')[-1]]] = peo_in_cluster[peo].get(
                        #         true_label[full_pic.split('/')[-1]], 0) + 1
    else:
        for peo, item in final_res.items():
            for full_pic in functools.reduce(operator.concat, functools.reduce(operator.concat, item.values())):
                if full_pic.split('/')[-3] in meeting_index:
                    if peo not in face_name:
                        face_name.append(peo)
                    face_in_cluster[face_name.index(peo)][meeting_index[full_pic.split('/')[-3]]] = 1
                if full_pic.split('/')[-1] in true_label:
                    peo_in_cluster[peo][true_label[full_pic.split('/')[-1]]] = peo_in_cluster[peo].get(true_label[full_pic.split('/')[-1]], 0) + 1


    # evaluation
    predict_peo = defaultdict(dict)
    precisions = []
    cnt2 = 0
    all = 0
    print('peo_in_cluster = ')
    with open(os.path.join(project_dir, 'peo_in_cluster.csv'), 'w') as f:
        print_color = {1: 'green', 0: 'red'}
        for peo, item in peo_in_cluster.items():
            for p, no in item.items():
                all += no
                if p == peo:
                    cnt2 += no
            major_element = max(item.items(), key=operator.itemgetter(1))[0]
            print(' \'' + colored(peo, print_color[peo == major_element]) + '\':')
            print('  {},'.format(item))
            if peo == major_element:
                precisions.append(item[major_element] / sum(item.values()))
            for part_peo, cnt in item.items():
                predict_peo[part_peo][peo] = cnt
                f.write('{}, {}, {}\n'.format(peo, part_peo, cnt))
    print('precisions: {}'.format(precisions))
    correct_cnt = 0
    correct_peo = []
    with open(os.path.join(project_dir, 'predict_peo.csv'), 'w') as f:
        for peo, item in predict_peo.items():
            major_element = max(item.items(), key=operator.itemgetter(1))[0]
            if major_element == peo:
                correct_peo.append(peo)
                correct_cnt += 1
            for part_peo, cnt in item.items():
                f.write('{}, {}, {}\n'.format(peo, part_peo, cnt))
    correct_peo.sort()
    print('precision = {}'.format(sum(precisions) / len(precisions)))
    print(colored('purity: {}'.format(cnt2 / all), 'yellow'))
    print('k = {}'.format(victims_thres))
    print('correct = {}'.format(correct_cnt))
    print('correct_peo = {}'.format(correct_peo))
    print('unpredicted_peo = {}'.format(set(poi_people) - set(correct_peo)))
    print(colored('choose {} correct: {} with all: {}'.format(victims_thres, correct_cnt, len(cols)), 'magenta'))

    return final_res, save_matrix, labeled


def associate(real_world, audio):
    project_dir = os.path.dirname(os.getcwd())
    cfg = utils.get_config(project_dir)
    data_path = os.path.join(project_dir, cfg['base_conf']['data_path'])
    wifi_thres = cfg['parameters']['wifi_threshold']
    meeting_thres = cfg['parameters']['meeting_threshold']
    cdist_metric = cfg['parameters']['cdist_metric']
    victims_thres = cfg['parameters']['estimated_victims']
    omega = cfg['parameters']['omega']
    #TODO: do not hard code

    # meeting_npy_paths, the npy path that contains feature vectors, after running feature_extraction
    meeting_npy_paths = utils.get_meeting_and_path(data_path, r'.+\.npy$')
    bio_path = utils.get_meeting_and_path(data_path, r'.+\.pk$')

    # get participants' names, given the corresponding WiFi sniffing files (wifi_thres is the cutoff rss threshold)
    meeting_poi_name = utils.get_meeting_poi_name(data_path, real_world, wifi_thres)

    # remove filter people
    for meeting, people_names in meeting_poi_name.items():
        meeting_poi_name[meeting] = [poi for poi in people_names if poi not in cfg['filter_people']['video']]

    # flatten to get unique poi name
    poi_people = sorted({name for people_names in meeting_poi_name.values() for name in people_names})
    logging.info('poi people: {}'.format(poi_people))

    meeting_name = sorted(meeting_poi_name.keys())
    meeting_num = len(meeting_poi_name)
    # get meeting index by meeting name
    meeting_index = dict(zip(meeting_name, list(range(meeting_num))))
    poi_num = len(poi_people)

    context_infor = np.zeros([meeting_num, poi_num]).astype(np.float64)
    for meeting, people_names in meeting_poi_name.items():
        for poi in people_names:
            context_infor[meeting_index[meeting], poi_people.index(poi)] = 1

    bio_info = np.empty([0, cfg['pre_process']['dimension']])

    bio_paths = []
    for meeting in bio_path.keys():
        with open(bio_path[meeting], 'rb') as f:
            bio_paths.extend(pickle.load(f))
        # iou vec is the face / voice features via feature extractor
        iou_vec = np.load(meeting_npy_paths[meeting])
        try:
            bio_info = np.vstack((bio_info, iou_vec))
        except:
            logging.error('error in numpy vstack: {}'.format(meeting))

    # construct mac attendance vector
    if real_world:
        real_mac_attendance = defaultdict(lambda: [0] * meeting_num)
        meeting_people_mac = utils.get_meeting_all_people_mac(data_path, real_world)

        for meeting, macs in meeting_people_mac.items():
            for mac in macs:
                real_mac_attendance[mac][meeting_index[meeting]] = 1

        # load and map poi name and mac address, respectively
        poi_name_mac = {}
        for mac, poi in cfg['mac_name']:
            if poi not in poi_name_mac:
                poi_name_mac[poi] = mac
            else:
                logging.warning('duplicate mac address of {} with {} and {}'.format(poi, poi_name_mac[poi], mac))

        # use mac_index to get the index of mac address in mac_attendance
        mac_index = {}
        poi_mac_attendance = []
        cnt = 0
        for poi in poi_people:
            if poi in poi_name_mac and poi_name_mac[poi] in real_mac_attendance:
                # divide real_mac_attendance into two sequences instead of random: first with poi and following non_poi part.
                poi_mac_attendance.append(real_mac_attendance[poi_name_mac[poi]])
                mac_index[cnt] = poi_name_mac[poi]
                cnt += 1
                real_mac_attendance.pop(poi_name_mac[poi])
            else:
                # check if all poi has the name-mac mapping and poi attend at least one meeting
                logging.error('poi {} do not have mac information or have not attended at least one meeting'.format(poi))

        # add the non poi part of mac_attendance
        non_poi_mac_attendance = []
        for mac, attendance in real_mac_attendance.items():
            mac_index[cnt] = mac
            cnt += 1
            non_poi_mac_attendance.append(attendance)

        mac_attendance = np.concatenate((poi_mac_attendance, non_poi_mac_attendance))

        # remove always-on mac address, detected over 90% of all meetings, which may be router or something.
        always_on_index = np.where(mac_attendance.sum(axis=1) > 0.9 * mac_attendance[1])[0]
        mac_attendance = np.delete(mac_attendance, always_on_index, axis=0)
        logging.info('always on mac address no: {} of all meeting no: {}'.format(len(always_on_index), mac_attendance.shape[1]))
        logging.info('mac attendance len: {} with poi no: {}'.format(len(mac_attendance), len(poi_mac_attendance)))

        # concatenate features and attendance vector (ctx information)
        assert len(bio_paths) == len(bio_info)
        logging.info('bio_paths and bio_info len: {}'.format(len(bio_paths)))

        # get bio_features in valid meeting
        if meeting_thres == -1:
            meeting_thres = meeting_num
        bio_features = []
        for bio_feature, path in zip(bio_info, bio_paths):
            parent = utils.get_parent_folder_name(path[0], 3)
            c = parent.split('_')
            if meeting_index['%s_%s' % (c[0], c[1])] < meeting_thres:
                bio_features.append(bio_feature)

        # event vector of MAC addresses
        wifi_people_in_meetings = np.zeros([poi_num, min(meeting_num, meeting_thres)])
        for i in range(poi_num):
            for name in meeting_name:
                if poi_people[i] in meeting_poi_name[name]:
                    if meeting_index[name] < meeting_thres:
                        wifi_people_in_meetings[i, meeting_index[name]] = 1

        scan(bio_features, bio_paths, bio_info, real_world, cdist_metric, victims_thres, omega, meeting_num, meeting_thres, meeting_index, wifi_people_in_meetings, poi_people, project_dir, audio)
