# 2021/10/20
# valのデータセットに対して、AttnMAPを生成するスクリプト

from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.utils import calc_tea_iou
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data
from datasets import get_real_polygons

from model import RNN_ENCODER, CNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image
import json
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def evaluate(dataloader, cnn_model, rnn_model, batch_size, image_dir, ixtoword, category_words_ix, category_real_polygons, category):
    MAX_GEN = 2000
    cnn_model.eval()
    rnn_model.eval()

    mean_iou_list = []
    std_iou_list = []
    # 実際にiouの計算が行われた文の数
    num_calc_iou = 0
    num_sentence = 0
    for step, data in enumerate(dataloader, 0):
        print(f'{step} / {len(dataloader)}')

        if step == MAX_GEN:
            break
        imgs, captions, cap_lens, \
                class_ids, keys = prepare_data(data)

        '''
        imgs[0]に入っている。shapeはtorch.Size([8, 3, 299, 299])。8の部分はbatch size
        captions shape = torch.Size([8, 18]) => つまり長いやつにあわしたpadding
        cap_lens = tensor([18, 12, 12, 11, 10, 10,  9,  7], device='cuda:0')
        keys = ['COCO_val2014_000000206851', 'COCO_val2014_000000548878', ...]
        '''

        words_features, sent_code = cnn_model(imgs[-1])
        att_sze = words_features.size(2)
        '''
        word_features: 単語レベルの特徴量, sent_code: 文章レベルの特徴量
        words_features= torch.Size([8, 256, 17, 17])
        sent_code= torch.Size([8, 256])
        att_sze= 17
        '''
        
        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)

        '''
        attnはlist型。中身はtensor。つまりattn=[tensor, tensor, ...]（今回の場合だと長さ8）
        attn[0] shape= torch.Size([1, 15, 17, 17])
        '''

        if category_words_ix:
            real_polygons_list = get_real_polygons(keys, category_real_polygons)
            mean_iou, std_iou, num_calc_iou_batch = calc_tea_iou(imgs[-1].cpu(), captions, ixtoword, attn, att_sze, category_words_ix, real_polygons_list)

            # categoryがある場合は正解ポリゴンの描画
            # img_set, sentences = \
            #     build_super_images(imgs[-1].cpu(), captions,
            #                         ixtoword, attn, att_sze,
            #                         category_words_ix=category_words_ix, real_polygons_list=real_polygons_list)

            # print(f'one batch mean tea-iou: {mean_iou}')
            # print(f'one batch num_calc_iou: {num_calc_iou_batch}')
            mean_iou_list.append(mean_iou)
            std_iou_list.append(std_iou)
            num_sentence += len(keys)
            num_calc_iou += num_calc_iou_batch

        else:
            img_set, sentences = \
                build_super_images(imgs[-1].cpu(), captions,
                                    ixtoword, attn, att_sze)

        # if img_set is not None:
        #     im = Image.fromarray(img_set)
        #     fullpath = '%s/attention_maps%d.png' % (image_dir, step)
        #     im.save(fullpath)

    mean_tea_iou = sum(mean_iou_list) / len(mean_iou_list)
    std_tea_iou = sum(std_iou_list) / len(std_iou_list)
    result_dic = {category: {'tea_iou': mean_tea_iou, 'std_tea_iou': std_tea_iou, 'num_calu_iou': num_calc_iou, 'num_sentence': num_sentence}}

    return result_dic


def build_models():
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '/data/Users/nakachi/stair/AttnGAN/output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    image_dir = os.path.join(output_dir, 'Image')
    # mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    # bboxのためにresizeのみにする
    image_transform = transforms.Compose([
        transforms.Resize((imsize, imsize))
        ])

    dataset = TextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # categoryごとのtea_iouが欲しいのでカテゴリー一覧
    category_words_dic_path = '/home/nakachi/data/category_split/category_words_ja_dic.json'
    with open(category_words_dic_path) as f:
        category_words_dic = json.load(f)

    result_dic = {}
    reslut_dic_save_path = '/home/nakachi/data/category_tea_iou_ja_20220112.pk'
    LANG = '_ja'
    for category in category_words_dic.keys():
        print(f'{category} process...')
        data_dir = f'/home/nakachi/data/category_split/{category}{LANG}/'
        print(data_dir)

        # # validation data #
        dataset_val = TextDataset(data_dir, 'val', # stair用。本来はtest
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform, category=category)

        dataloader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size, drop_last=True,
            num_workers=int(cfg.WORKERS))

        text_encoder, image_encoder, labels, start_epoch = build_models()


        try:
            category_result_dic = evaluate(dataloader_val, image_encoder, text_encoder, batch_size, image_dir, dataset_val.ixtoword, dataset_val.category_words_ix, dataset_val.category_real_polygons, category)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        result_dic.update(category_result_dic)
        with open(reslut_dic_save_path, mode="wb") as f:
            pickle.dump(result_dic, f)