import time
import torch
import os
import cv2
import Config
import alphabets
from torch.autograd import Variable
import lib.convert
import lib.dataset
from PIL import Image
import Net.net as Net
import numpy as np
from math import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

root_recs = 'data/cutphoto/save/'

crnn_model_path = '../weights/crnn/netCRNN_4_48000.pth'
IMG_ROOT = '../data/crnn'
running_mode = 'gpu'
alphabet = alphabets.alphabet
nclass = len(alphabet) + 1


def dumpRotateImage(img, rec):
    xDim, yDim = img.shape[1], img.shape[0]

    # fixme 扩展文字白边 参数为经验值
    xlength = int((rec[4] - rec[0]) * 0.02)

    ylength = int((rec[5] - rec[1]) * 0.05)

    pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
    pt2 = (rec[6], rec[7])
    pt3 = (min(rec[4] + xlength, xDim - 2),
           min(yDim - 2, rec[5] + ylength))

    degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # fixme 图像倾斜角度

    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation,
                                  np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation,
                                  np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]

    pt1_N = []
    pt3_N = []
    pt1_N.append(max(1, int(pt1[0])))
    pt1_N.append(max(1, int(pt1[1])))
    pt3_N.append(min(xdim - 1, int(pt3[0])))
    pt3_N.append(min(ydim - 1, int(pt3[1])))

    imgRotation = np.uint8(imgRotation)
    img_rot = Image.fromarray(imgRotation)
    img_rec = img_rot.crop((pt1_N[0], pt1_N[1], pt3_N[0], pt3_N[1]))

    return img_rec


def crnn_recognition(cropped_image, model):
    # 标签转换
    converter = lib.convert.strLabelConverter(alphabet)

    # 图像灰度化
    image = cropped_image.convert('L')
    w = int(image.size[0] / (280 * 1.0 / Config.infer_img_w))
    transformer = lib.dataset.resizeNormalize((w, Config.img_height))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))

    # 预测输出，解码成文字
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))


def predict_text(model, recs_all, recs_len, img_all, img_name=None):
    converter = lib.convert.strLabelConverter(alphabet)

    if not os.path.exists(root_recs):
        os.makedirs(root_recs)

    img_index = 0
    result = ''

    # fixme 当前是前面所有长度的和
    for i in range(len(recs_len)):
        if i > 0:
            recs_len[i] += recs_len[i - 1]

    for i in range(len(recs_all)):
        for j in range(len(recs_len)):
            if i < recs_len[j]:
                img_index = j
                break

        # 根据坐标旋转图片
        img_rec = dumpRotateImage(img_all[img_index], recs_all[i]).convert('L')

        # img_rec.show()
        image = img_rec.convert('L')

        w = int(image.size[0] / (280 * 1.0 / Config.infer_img_w))
        transformer = lib.dataset.resizeNormalize((w, Config.img_height))
        image = transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        model.eval()
        preds = model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))

        # 预测输出，解码成文字
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        result += (format(sim_pred) + '\n')

    return result


if __name__ == '__main__':

    model = Net.CRNN(nclass)
    if running_mode == 'gpu' and torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(crnn_model_path))
    else:
        model.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))

    print('loading pretrained model from {0}'.format(crnn_model_path))

    files = sorted(os.listdir(IMG_ROOT))
    for file in files:
        started = time.time()
        full_path = os.path.join(IMG_ROOT, file)
        print("=============================================")
        print("ocr image is %s" % full_path)
        image = Image.open(full_path)

        crnn_recognition(image, model)
        finished = time.time()
        print('elapsed time: {0}'.format(finished - started))
