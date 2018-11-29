import sys
import json
import os
import glob
from os.path import join as fullfile
import numpy as np


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def compute_success_overlap(gt_bb, result_bb):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = overlap_ratio(gt_bb, result_bb)
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success


def compute_success_error(gt_center, result_center):
    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success


def get_result_bb(arch, seq):
    result_path = fullfile(arch, seq + '.txt')
    temp = np.loadtxt(result_path, delimiter=',').astype(np.float)
    return np.array(temp)


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def eval_auc(dataset='OTB2015', tracker_reg='S*', start=0, end=1e6):
    list_path = os.path.join('data', dataset + '.json')
    annos = json.load(open(list_path, 'r'))
    seqs = list(annos.keys())

    OTB2013 = ['carDark', 'car4', 'david', 'david2', 'sylvester', 'trellis', 'fish', 'mhyang', 'soccer', 'matrix',
               'ironman', 'deer', 'skating1', 'shaking', 'singer1', 'singer2', 'coke', 'bolt', 'boy', 'dudek',
               'crossing', 'couple', 'football1', 'jogging_1', 'jogging_2', 'doll', 'girl', 'walking2', 'walking',
               'fleetface', 'freeman1', 'freeman3', 'freeman4', 'david3', 'jumping', 'carScale', 'skiing', 'dog1',
               'suv', 'motorRolling', 'mountainBike', 'lemming', 'liquor', 'woman', 'faceocc1', 'faceocc2',
               'basketball', 'football', 'subway', 'tiger1', 'tiger2']

    trackers = glob.glob(fullfile('test', dataset, tracker_reg))
    trackers = trackers[start:min(end, len(trackers))]

    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    # thresholds_error = np.arange(0, 51, 1)

    success_overlap = np.zeros((n_seq, len(trackers), len(thresholds_overlap)))
    # success_error = np.zeros((n_seq, len(trackers), len(thresholds_error)))
    for i in range(n_seq):
        seq = seqs[i]
        gt_rect = np.array(annos[seq]['gt_rect']).astype(np.float)
        gt_center = convert_bb_to_center(gt_rect)
        for j in range(len(trackers)):
            tracker = trackers[j]
            print('{:d} processing:{} tracker: {}'.format(i, seq, tracker))
            bb = get_result_bb(tracker, seq)
            center = convert_bb_to_center(bb)
            success_overlap[i][j] = compute_success_overlap(gt_rect, bb)
            # success_error[i][j] = compute_success_error(gt_center, center)

    print('Success Overlap')

    if 'OTB2015' == dataset:
        OTB2013_id = []
        for i in range(n_seq):
            if seqs[i] in OTB2013:
                OTB2013_id.append(i)
        max_auc_OTB2013 = 0.
        max_name_OTB2013 = ''
        for i in range(len(trackers)):
            auc = success_overlap[OTB2013_id, i, :].mean()
            if auc > max_auc_OTB2013:
                max_auc_OTB2013 = auc
                max_name_OTB2013 = trackers[i]
            print('%s(OTB2013 AUC:%.4f)' % (trackers[i], auc))

        max_auc = 0.
        max_name = ''
        for i in range(len(trackers)):
            auc = success_overlap[:, i, :].mean()
            if auc > max_auc:
                max_auc = auc
                max_name = trackers[i]
            print('%s(OTB2015 AUC:%.4f)' % (trackers[i], auc))

        print('\nOTB2013 Best: %s(%.4f)' % (max_name_OTB2013, max_auc_OTB2013))
        print('\nOTB2015 Best: %s(%.4f)' % (max_name, max_auc))
    else:
        max_auc = 0.
        max_name = ''
        for i in range(len(trackers)):
            auc = success_overlap[:, i, :].mean()
            if auc > max_auc:
                max_auc = auc
                max_name = trackers[i]
            print('%s(%.4f)' % (trackers[i], auc))

        print('\n%s Best: %s(%.4f)' % (dataset, max_name, max_auc))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('python eval_otb.py OTB2015 Siam* 0 10')
        exit()
    dataset = sys.argv[1]
    tracker_reg = sys.argv[2]
    start = int(float(sys.argv[3]))
    end = int(float(sys.argv[4]))
    eval_auc(dataset, tracker_reg, start, end)
