import os
import sys
import json
import os.path as osp
import numpy as np
import xmltodict
from tqdm import tqdm
import math

sys.path.append("..")

def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
            + cal_line_length(combinate[i][1], dst_coordinate[1]) \
            + cal_line_length(combinate[i][2], dst_coordinate[2]) \
            + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)

def rbox2poly_single(rrect):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width/2, -height/2, width/2, height/2
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)
    return poly

def parse_ann_info(objects):
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    # only one annotation
    if type(objects) != list:
        objects = [objects]
    for obj in objects:
        if obj['difficult'] == '0':
            bbox = float(obj['mbox_cx']), float(obj['mbox_cy']), float(
                obj['mbox_w']), float(obj['mbox_h']), float(obj['mbox_ang'])
            label = 'ship'
            bboxes.append(bbox)
            labels.append(label)
        elif obj['difficult'] == '1':
            bbox = float(obj['mbox_cx']), float(obj['mbox_cy']), float(
                obj['mbox_w']), float(obj['mbox_h']), float(obj['mbox_ang'])
            label = 'ship'
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
    return bboxes, labels, bboxes_ignore, labels_ignore

def parse_ann_info_C6(objects):
    CLASSES_map = {'100000001':'merchant ship', '100000002':'aircraft carrier', '100000003':'destroyers', '100000004':'merchant ship', '100000005':'aircraft carrier', '100000006':'aircraft carrier',
                '100000007':'destroyers', '100000008':'destroyers', '100000009':'destroyers', '100000010':'transport dock', '100000011':'destroyers', '100000012':'aircraft carrier',
                '100000013':'aircraft carrier', '100000014':'destroyers', '100000015':'transport dock', '100000016':'aircraft carrier', '100000017':'destroyers', '100000018':'merchant ship',
                '100000019':'transport dock', '100000020':'merchant ship', '100000022':'merchant ship', '100000024':'merchant ship', '100000025':'merchant ship', '100000026':'merchant ship',
                '100000027':'submarine', '100000028':'destroyers', '100000029':'medical ship', '100000030':'merchant ship', '100000031':'aircraft carrier', '100000032':'aircraft carrier', '100000033':'aircraft carrier'}

    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    # only one annotation
    if type(objects) != list:
        objects = [objects]
    for obj in objects:
        if obj['difficult'] == '0':
            bbox = float(obj['mbox_cx']), float(obj['mbox_cy']), float(
                obj['mbox_w']), float(obj['mbox_h']), float(obj['mbox_ang'])
            label = CLASSES_map[obj['Class_ID']]
            bboxes.append(bbox)
            labels.append(label)
        elif obj['difficult'] == '1':
            bbox = float(obj['mbox_cx']), float(obj['mbox_cy']), float(
                obj['mbox_w']), float(obj['mbox_h']), float(obj['mbox_ang'])
            label = CLASSES_map[obj['Class_ID']]
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
    return bboxes, labels, bboxes_ignore, labels_ignore


def ann_to_txt(ann):
    out_str = ''
    for bbox, label in zip(ann['bboxes'], ann['labels']):
        poly = rbox2poly_single(bbox)
        str_line = '{} {} {} {} {} {} {} {} {} {}\n'.format(
            poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], label, '0')
        out_str += str_line
    for bbox, label in zip(ann['bboxes_ignore'], ann['labels_ignore']):
        poly = rbox2poly_single(bbox)
        str_line = '{} {} {} {} {} {} {} {} {} {}\n'.format(
            poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], label, '1')
        out_str += str_line
    return out_str


def generate_txt_labels(root_path):
    img_path = osp.join(root_path, 'images')
    label_path = osp.join(root_path, 'labelTxt')
    label_txt_path = osp.join(root_path, 'annotations_C6')
    # if not osp.exists(label_txt_path):
    #     os.mkdir(label_txt_path)

    img_names = [osp.splitext(img_name.strip())[0] for img_name in os.listdir(img_path)]
    pbar = tqdm(img_names)
    for img_name in pbar:
        pbar.set_description("HRSC2016 Preparation...")

        label = osp.join(label_path, img_name + '.xml')
        label_txt = osp.join(label_txt_path, img_name + '.txt')
        f_label = open(label)
        data_dict = xmltodict.parse(f_label.read())
        data_dict = data_dict['HRSC_Image']
        f_label.close()
        label_txt_str = ''
        # with annotations
        if data_dict['HRSC_Objects']:
            objects = data_dict['HRSC_Objects']['HRSC_Object']
            bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info_C6(
                objects)
            ann = dict(
                bboxes=bboxes,
                labels=labels,
                bboxes_ignore=bboxes_ignore,
                labels_ignore=labels_ignore)
            label_txt_str = ann_to_txt(ann)
        with open(label_txt, 'w') as f_txt:
            f_txt.write(label_txt_str)


if __name__ == '__main__':
    # generate_txt_labels('/project/jmhan/data/HRSC2016/Train')
    generate_txt_labels('datasets/HRSC2016/Test')
    print('done!')
