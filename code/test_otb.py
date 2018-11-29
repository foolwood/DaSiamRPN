# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import argparse, cv2, torch, json
import numpy as np
from os import makedirs
from os.path import realpath, dirname, join, isdir, exists

from net import SiamRPNotb
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import rect_2_cxy_wh, cxy_wh_2_rect

parser = argparse.ArgumentParser(description='PyTorch SiamRPN OTB Test')
parser.add_argument('--dataset', dest='dataset', default='OTB2015', help='datasets')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')


def track_video(model, video):
    toc, regions = 0, []
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)  # TODO: batch load
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos, target_sz = rect_2_cxy_wh(gt[f])
            state = SiamRPN_init(im, target_pos, target_sz, model)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(gt[f])
        elif f > 0:  # tracking
            state = SiamRPN_track(state, im)  # track
            location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])
            regions.append(location)
        toc += cv2.getTickCount() - tic

        if args.visualization and f >= 0:  # visualization
            if f == 0: cv2.destroyAllWindows()
            if len(gt[f]) == 8:
                cv2.polylines(im, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            else:
                cv2.rectangle(im, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            if len(location) == 8:
                cv2.polylines(im, [location.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]  #
                cv2.rectangle(im, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(video['name'], im)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # save result
    video_path = join('test', args.dataset, 'SiamRPN_AlexNet_OTB2015')
    if not isdir(video_path): makedirs(video_path)
    result_path = join(video_path, '{:s}.txt'.format(video['name']))
    with open(result_path, "w") as fin:
        for x in regions:
            fin.write(','.join([str(i) for i in x])+'\n')

    print('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f / toc))
    return f / toc


def load_dataset(dataset):
    base_path = join(realpath(dirname(__file__)), 'data', dataset)
    if not exists(base_path):
        print("Please download OTB dataset into `data` folder!")
        exit()
    json_path = join(realpath(dirname(__file__)), 'data', dataset + '.json')
    info = json.load(open(json_path, 'r'))
    for v in info.keys():
        path_name = info[v]['name']
        info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
        info[v]['gt'] = np.array(info[v]['gt_rect'])-[1,1,0,0]  # our tracker is 0-index
        info[v]['name'] = v
    return info


def main():
    global args, v_id
    args = parser.parse_args()

    net = SiamRPNotb()
    net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNOTB.model')))
    net.eval().cuda()

    dataset = load_dataset(args.dataset)
    fps_list = []
    for v_id, video in enumerate(dataset.keys()):
        fps_list.append(track_video(net, dataset[video]))
    print('Mean Running Speed {:.1f}fps'.format(np.mean(np.array(fps_list))))


if __name__ == '__main__':
    main()
