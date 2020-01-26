import os
import cv2
import logging
import utils
import pickle
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
from . import facenet, utils
from sklearn.cluster import AgglomerativeClustering


def feature_extraction(real_world: bool):
    project_dir = os.path.dirname(os.getcwd())
    cfg = utils.get_config(project_dir)
    data_path = os.path.join(project_dir, cfg['base_conf']['data_path'])
    wifi_thres = cfg['pre_process']['wifi_threshold']

    if cfg['specs']['set_gpu']:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['specs']['gpu_num'])

    meeting_poi_name = utils.get_meeting_poi_name(data_path, real_world, wifi_thres)

    # remove meeting that no one attended
    for meeting, people_names in meeting_poi_name.items():
        if len(people_names) == 0:
            shutil.rmtree(os.path.join(data_path, meeting))

    # multiple runs
    # remove classifier and npy file generated last time
    for meeting in os.listdir(data_path):
        classifier_path = os.path.join(project_dir, data_path, meeting, 'classifier')
        if os.path.exists(classifier_path):
            shutil.rmtree(classifier_path)
    meeting_npy_paths = utils.get_meeting_and_path(data_path, r'.+\.npy')
    for npy_path in meeting_npy_paths:
        if os.path.exists(meeting_npy_paths[npy_path]):
            os.remove(meeting_npy_paths[npy_path])

    meeting_paths = [item for item in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, item))]
    meeting_paths = list(map(lambda x: os.path.join(data_path, x), meeting_paths))
    meeting_people_num = utils.get_meeting_poi_num(data_path, real_world, wifi_thres)

    with tf.Graph().as_default():
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.visible_device_list = str(cfg['specs']['gpu_num'])
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            facenet.load_model(os.path.join(project_dir, 'models', cfg['pre_process']['model_name']))
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            for meeting_path in meeting_paths:
                mtcnn_path = os.path.join(meeting_path, cfg['base_conf']['mtcnn_path'])
                pic_paths = os.listdir(mtcnn_path)
                pic_paths.sort(key=lambda x: utils.get_iou_num(x, real_world))
                meeting = utils.get_parent_folder_name(meeting_path, 1)
                image_list = []
                result = np.empty([0, cfg['pre_process']['dimension']])
                piece_num = cfg['pre_process']['piece_num']
                if len(pic_paths) == 0:
                    shutil.rmtree(meeting_path)
                    continue
                for pic in pic_paths:
                    imm = Image.open(os.path.join(mtcnn_path, pic))
                    try:
                        imm.verify()
                    except Exception:
                        logging.error('invalid image: {}'.format(os.path.join(mtcnn_path, pic)))
                    im = cv2.imread(os.path.join(mtcnn_path, pic))
                    im = cv2.resize(im, (160, 160))
                    prewhitened = facenet.prewhiten(im)
                    image_list.append(prewhitened)
                    if len(image_list) == piece_num:
                        images = np.stack(image_list)
                        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        result = np.vstack((result, emb))
                        image_list.clear()

                if len(image_list) != 0:
                    try:
                        images = np.stack(image_list)
                    except:
                        logging.error('may not resize the image')
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    result = np.vstack((result, emb))

                features = result.tolist()

                pre_iou_num = utils.get_iou_num(pic_paths[0], real_world)
                same_iou_vec = np.empty([0, cfg['pre_process']['dimension']])
                iou_vec = []
                iou_pic_path = []
                single_iou_path = []
                for name, vec in zip(pic_paths, features):
                    iou_num = utils.get_iou_num(name, real_world)
                    if iou_num == pre_iou_num:
                        same_iou_vec = np.vstack((same_iou_vec, vec))
                        single_iou_path.append(os.path.join(meeting_path, 'mtcnn', name))
                    else:
                        iou_vec.append(np.mean(same_iou_vec, axis=0))
                        iou_pic_path.append(single_iou_path)
                        single_iou_path = []
                        same_iou_vec = np.empty([0, cfg['pre_process']['dimension']])
                        same_iou_vec = np.vstack((same_iou_vec, vec))
                        single_iou_path.append(os.path.join(meeting_path, 'mtcnn', name))
                        pre_iou_num = iou_num
                if len(single_iou_path) != 0:
                    iou_vec.append(np.mean(same_iou_vec, axis=0))
                    iou_pic_path.append(single_iou_path)

                with open(os.path.join(meeting_path, 'pics%s.pk' % meeting), 'wb') as f:
                    pickle.dump(iou_pic_path, f)
                np.save(os.path.join(meeting_path, 'vec%s.npy' % meeting), iou_vec)

                classifier_path = os.path.join(meeting_path, 'classifier')
                if os.path.exists(classifier_path):
                    shutil.rmtree(classifier_path)
                os.mkdir(classifier_path)
                for i in range(meeting_people_num[str(meeting)]):
                    os.makedirs(os.path.join(classifier_path, str(i)))

                if len(iou_vec) == 0:
                    continue
                if len(iou_vec) < 2:
                    continue
                if len(iou_vec) > meeting_people_num[meeting]:
                    cluster_number = meeting_people_num[meeting]
                else:
                    cluster_number = len(iou_vec)
                kmeans = AgglomerativeClustering(n_clusters=cluster_number, linkage='average').fit(iou_vec)

                index = 0
                for d in kmeans.labels_:
                    name = utils.get_parent_folder_name(iou_pic_path[index][0], 1)
                    shutil.copyfile(iou_pic_path[index][0], os.path.join(meeting_path, 'classifier', str(d), name))
                    index += 1
