import os
import errno
import numpy as np
from statistics import mean as sta_mean
from statistics import stdev as sta_stdev
from torch.nn import init

import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform
import cv2
from shapely.geometry import Polygon


from miscc.config import cfg


# For visualization ################################################
COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a font
    # fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/fonts-japanese-gothic.ttf', 50)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('utf-8', 'ignore').decode('utf-8')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM,
                       category_words_ix=None, real_polygons_list=None):
    nvis = 8
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        # ここの処理は通らない
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        # ここの処理は通る
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]

    # 本当はreal_imgs=torch.Size([3, 299, 299])だがreal_imgs torch.Size([3, 272, 272])に直す
    real_imgs = \
        nn.functional.interpolate(real_imgs,size=(vis_size, vis_size),
                                  mode='bilinear', align_corners=False)
    # [-1, 1] --> [0, 1]: この処理は多分、画像は-1からスタートできないから0からに直している
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        # ここの処理は通らない
        lr_imgs = \
            nn.functional.interpolate(lr_imgs,size=(vis_size, vis_size),
                                  mode='bilinear', align_corners=False)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    # -> ここは18. predict_stair.ymlに記述がない場合、configから読まれるから
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        # 一文ごとのloop
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        cap = captions[i].data.cpu().numpy()

        # -> attn_maps= torch.Size([1, 8(単語数？), 17, 17])
        # --> 1 x 1 x 17 x 17 : と書かれているが実際はtorch.Size([1, 8(単語数？), 17, 17])
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        # -> torch.Size([1, 9(単語数？ + 1), 17, 17])
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]
            if (vis_size // att_sze) > 1:
                # この処理は通る
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze,
                                                     multichannel=True)
                # -> この時点ではone_map= (272, 272, 3)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            # 単語ごとのloop
            MODE = 'polygon' # bbox or polygon
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255
                # -> この時点でもone_map= (272, 272, 3)

                if MODE == 'bbox':
                    # attentionよりbboxを決める
                    gray_one_map = one_map[:,:,1]
                    mean =  np.mean(one_map)
                    over_mean_index = np.where(gray_one_map > mean + (mean / 2))
                    if over_mean_index[0].any():
                        y_min_bbox = np.min(over_mean_index[0])
                        y_max_bbox = np.max(over_mean_index[0])
                        x_min_bbox = np.min(over_mean_index[1])
                        x_max_bbox = np.max(over_mean_index[1])
                    else:
                        x_min_bbox = 0
                        x_max_bbox = 1
                        y_min_bbox = 0
                        y_max_bbox = 1
                elif MODE == 'polygon':
                    gray_one_map = one_map[:,:,1]   # one_mapには同じ値が入っているので一つだけ取り出す
                    mean = np.mean(one_map)
                    over_mean_bi_map = np.where(gray_one_map > mean + (mean / 2), 255, 0)

                PIL_im = Image.fromarray(np.uint8(img))
                if MODE == 'polygon':
                    PIL_att = Image.fromarray(np.uint8(over_mean_bi_map))
                else:
                    PIL_att = Image.fromarray(np.uint8(one_map))
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                if MODE == 'bbox':
                    # bboxの描画
                    bbox_draw = ImageDraw.Draw(merged)
                    bbox_draw.rectangle([x_min_bbox, y_min_bbox, x_max_bbox, y_max_bbox], outline=(255, 0, 0), width=3)
                elif MODE == 'polygon':
                    if j < len(cap)-1:
                        if cap[j-1] in category_words_ix:
                            # 正解ポリゴンの描画
                            # print(ixtoword[cap[j-1]].encode('utf-8', 'ignore').decode('utf-8'))
                            # over_mean_bi_mapからオブジェクトの輪郭を検出する
                            contours, _ = cv2.findContours(pil2cv(PIL_att), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            # 小さい輪郭は誤検出として削除する
                            contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
                            # attention mapの描画
                            attn_polygon_draw = ImageDraw.Draw(merged)
                            for cnt in contours:
                                flatten_cnt = cnt.flatten().tolist()
                                attn_polygon_draw.line(flatten_cnt, fill=(255, 255, 0), width=4)
                                attn_polygon_draw.polygon(flatten_cnt, outline=(255, 255,0))


                            ref_polygon_draw = ImageDraw.Draw(merged)
                            for real_polygon in real_polygons_list[i]:
                                # 正解ポリゴンの描画
                                ref_polygon_draw.line(real_polygon, fill=(255, 0, 0), width=4)
                                ref_polygon_draw.polygon(real_polygon, outline=(255, 0, 0))
                                # TEA-IoUの計算
                                attn_map_ious = []
                                ref_polygon = Polygon(np.array(real_polygon).reshape(-1, 2).tolist())
                                if not ref_polygon.is_valid:
                                    print('ref polygon is invalid')
                                    continue
                                for cnt in contours:
                                    # attentionごとのloop
                                    attn_polygon = Polygon(np.array(cnt).reshape(-1, 2).tolist())
                                    if not attn_polygon.is_valid:
                                        print('attn polygon is invalid')
                                        continue

                                    intersect = ref_polygon.intersection(attn_polygon).area
                                    union = ref_polygon.union(attn_polygon).area
                                    iou = intersect / union
                                    attn_map_ious.append(iou)

                                # maxのiouを入れる
                                if attn_map_ious and False:
                                    max_tea_iou = (max(attn_map_ious))
                                    # max tea iouの描画。テキストで
                                    x_list = real_polygon[0::2]
                                    y_list = real_polygon[1::2]
                                    x_center = min(x_list) + ((max(x_list) - min(x_list)) / 2)
                                    y_center = min(y_list) + ((max(y_list) - min(y_list)) / 2)
                                    txpos = (x_center, y_center)
                                    font = ImageFont.truetype("DejaVuSans.ttf", size=22)
                                    text = f'{max_tea_iou:.03f}'
                                    txw, _ = ref_polygon_draw.textsize(text, font=font)
                                    ref_polygon_draw.rectangle([txpos, (x_center+txw, y_center+22)], fill=(255, 0, 0))
                                    ref_polygon_draw.text(txpos, text, font=font, fill=(255, 255, 255))


                    else:
                        print('exceed cap len')

                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def build_super_images2(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, vis_size=256, topK=5):
    batch_size = real_imgs.size(0)
    max_word_num = np.max(cap_lens)
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                           dtype=np.uint8)

    real_imgs = \
        nn.functional.interpolate(real_imgs,size=(vis_size, vis_size),
                                    mode='bilinear', align_corners=False)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size, off1=0)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2./float(num_attn)
        #
        img = real_imgs[i]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []
        for j in range(num_attn):
            one_map = attn[j]
            mask0 = one_map > (2. * thresh)
            conf_score.append(np.sum(one_map * mask0))
            mask = one_map > thresh
            one_map = one_map * mask
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze,
                                                     multichannel=True)
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map)
        sorted_indices = np.argsort(conf_score)[::-1]

        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255
            #
            PIL_im = Image.fromarray(np.uint8(img))
            PIL_att = Image.fromarray(np.uint8(one_map))
            merged = \
                Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
            mask = Image.new('L', (vis_size, vis_size), (180))  # (210)
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3]

            row.append(np.concatenate([one_map, middle_pad], 1))
            #
            row_merge.append(np.concatenate([merged, middle_pad], 1))
            #
            txt = text_map[i * FONT_MAX:(i + 1) * FONT_MAX,
                           j * (vis_size + 2):(j + 1) * (vis_size + 2), :]
            row_txt.append(txt)
        # reorder
        row_new = []
        row_merge_new = []
        txt_new = []
        for j in range(num_attn):
            idx = sorted_indices[j]
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print('Warnings: txt', txt.shape, 'row', row.shape,
                  'row_merge_new', row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


####################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

####################################################################
def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def calc_tea_iou(real_imgs, captions, ixtoword, attn_maps, att_sze, category_words_ix, real_polygons_list):
    '''
    Text Embedding Attention-IoU
    前処理はbuild_super_images関数と一緒
    '''
    seq_len = 18
    vis_size = att_sze * 16
    # 本当はreal_imgs=torch.Size([3, 299, 299])だがreal_imgs torch.Size([3, 272, 272])に直す
    real_imgs = \
        nn.functional.interpolate(real_imgs,size=(vis_size, vis_size),
                                  mode='bilinear', align_corners=False)

    # [-1, 1] --> [0, 1]: この処理は多分、画像は-1からスタートできないから0からに直している
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))

    all_iou_list = []
    num_calc_iou_batch = 0
    for i in range(len(captions)):
        # 一文ごとのloop
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        cap = captions[i].data.cpu().numpy()

        # この文の何かしらの単語で、実際にIoUが行われたか？
        is_calc_iou = False

        # -> attn_maps= torch.Size([1, 8(単語数？), 17, 17])
        # --> 1 x 1 x 17 x 17 : と書かれているが実際はtorch.Size([1, 8(単語数？), 17, 17])
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        # -> torch.Size([1, 9(単語数？ + 1), 17, 17])
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]

        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]
            if (vis_size // att_sze) > 1:
                # この処理は通る
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze,
                                                     multichannel=True)
                # -> この時点ではone_map= (272, 272, 3)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            # 単語ごとのloop
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255

                gray_one_map = one_map[:,:,1]
                mean = np.mean(one_map)
                over_mean_bi_map = np.where(gray_one_map > mean + (mean / 2), 255, 0)
                PIL_att = Image.fromarray(np.uint8(over_mean_bi_map))

                if j < len(cap)-1:
                    if cap[j-1] in category_words_ix:
                        is_calc_iou = True
                        # print(ixtoword[cap[j-1]].encode('utf-8', 'ignore').decode('utf-8'))
                        # over_mean_bi_mapからオブジェクトの輪郭を検出する
                        contours, _ = cv2.findContours(pil2cv(PIL_att), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # 小さい輪郭は誤検出として削除する
                        contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

                        # 正解のpolygon数だけ入る
                        ious = []
                        for real_polygon in real_polygons_list[i]:
                            # 正解ポリゴンの描画
                            attn_map_ious = []
                            ref_polygon = Polygon(np.array(real_polygon).reshape(-1, 2).tolist())
                            if not ref_polygon.is_valid:
                                print('ref polygon is invalid')
                                continue
                            for cnt in contours:
                                # attentionごとのloop
                                attn_polygon = Polygon(np.array(cnt).reshape(-1, 2).tolist())
                                if not attn_polygon.is_valid:
                                    print('attn polygon is invalid')
                                    continue

                                intersect = ref_polygon.intersection(attn_polygon).area
                                union = ref_polygon.union(attn_polygon).area
                                iou = intersect / union
                                attn_map_ious.append(iou)

                            # maxのiouを入れる
                            if attn_map_ious:
                                ious.append(max(attn_map_ious))

                        all_iou_list.extend(ious)
        if is_calc_iou:
            num_calc_iou_batch += 1

    # 全てのiouの平均を返す
    mean_iou = sta_mean(all_iou_list) if len(all_iou_list) > 0 else 0
    std_iou = sta_stdev(all_iou_list) if len(all_iou_list) > 1 else 0
    return mean_iou, std_iou, num_calc_iou_batch
