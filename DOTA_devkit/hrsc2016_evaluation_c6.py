# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import matplotlib.pyplot as plt
import pylab as pl
# import cPickle
import numpy as np
import os
import polyiou
import xml.etree.ElementTree as ET
from functools import partial
import pdb
# 设置显示中文字体
#coding:utf-8
from matplotlib.font_manager import FontProperties

 

def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    # if not os.path.isdir(cachedir):
    #   os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # print('imagenames: ', imagenames)
    # if not os.path.isfile(cachefile):
    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        # print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))
        # if i % 100 == 0:
        #   print ('Reading annotation for {:d}/{:d}'.format(
        #      i + 1, len(imagenames)) )
        # save
        # print ('Saving cached annotations to {:s}'.format(cachefile))
        # with open(cachefile, 'w') as f:
        #   cPickle.dump(recs, f)
    # else:
    # load
    # with open(cachefile, 'r') as f:
    #   recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        # difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        difficult = np.array([0 for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0].split('.')[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    # print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    # print('check sorted_scores: ', sorted_scores)
    # print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    # print('check imge_ids: ', image_ids)
    # print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    print('check fp:', fp)
    print('check tp', tp)

    print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def main():
    detpath = r'work_dirs/ReDet_re50_refpn_3x_hrsc2016_C6/Task1_results/Task1_{:s}.txt'
    annopath = r'datasets/HRSC2016/Test/annotations_C6/{:s}.txt'  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = r'datasets/HRSC2016/Test/test.txt'

    # classnames = ['merchant ship', 'aircraft carrier', 'destroyers', 'transport dock', 'submarine', 'medical ship']
    classnames = ['merchant', 'aircraft', 'destroyers', 'transport', 'submarine', 'medical']
    classname_c = ['商船','航母','驱逐舰','运输舰','潜艇','医疗船']
    colors_cnames = ['#0000FF', '#FF8C00', '#008000', '#FF0000', '#800080', \
                        '#A52A2A', '#FFC0CB', '#FFFF00', '#808000', '#00FFFF', \
                        '#800000', '#8A2BE2', '#008B8B', '#CD853F', '#FF00FF']
    classaps = []
    map = 0
    results = []
    for i, classname in enumerate(classnames):
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.3, use_07_metric=True)
        map = map + ap
        print('ap: ', ap)
        classaps.append(ap)

        if classname == 'medical':
            classname = 'merchant'#'商船'
        elif classname == 'transport':
            classname = 'transport'#'运输船'
        elif classname == 'destroyers':
            classname = 'submarine'#'潜艇'
        elif classname == 'submarine':
            classname = 'medical'#'医疗船'
        elif classname == 'merchant':
            classname = 'destroyers'#'驱逐舰'
        elif classname == 'aircraft':
            classname = 'aircraft'#'航母'

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # zhfont = FontProperties(fname='/opt/conda/lib/python3.8/site-packages/matplotlib-3.6.0rc2-py3.8-linux-x86_64.egg/matplotlib/mpl-data/fonts/ttf/DejaVuSerif.ttf')
        pl.plot(rec, prec, color = colors_cnames[i], lw=2, label=classname_c[i])
        results.append([rec, prec, classname])
        pl.xlabel('召回率')
        pl.ylabel('精确率')
        plt.grid(True)
        pl.ylim([0.0, 1.05])
        pl.xlim([0.0, 1.0])
    map = map/len(classnames)

    # pl.title('Precision-Recall')
    # pl.title('虚警率')
    pl.legend(loc="lower left", fontsize = 'xx-small')  
    plt.savefig('work_dirs/ReDet_re50_refpn_3x_hrsc2016_C6/P-R_r_c.png', dpi=200)

    print('map:', map)

    # aps = []
    # for iou_thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    #     rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, 'ship', ovthresh=iou_thr, use_07_metric=True)
    #     aps.append(ap * 100)

    # ap_50 = aps[0]
    # ap_75 = aps[5]
    # map = np.array(aps).mean()

    # print('AP50: {:.2f}\tAP75: {:.2f}\t mAP: {:.2f}'.format(ap_50, ap_75, map))


if __name__ == '__main__':
    main()
