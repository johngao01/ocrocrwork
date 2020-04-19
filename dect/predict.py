import argparse

import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import cfg
from label import point_inside_of_quad
from network import East
from preprocess import resize_image
from nms import nms

# root_temp = 'demo/temp'
# root_predict = 'demo/predict'


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_path + '_subim%d.jpg' % s)


def predict(east_detect, img_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    # 将张量的值 调到【-1 1】
    img = preprocess_input(img, mode='tf')
    # 变成4维张量
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    with Image.open(img_path) as im:
        im_array = image.img_to_array(im.convert('RGB'))
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        draw = ImageDraw.Draw(im)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            px = (j + 0.5) * cfg.pixel_size
            py = (i + 0.5) * cfg.pixel_size
            line_width, line_color = 1, 'red'
            if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                if y[i, j, 2] < cfg.trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                    line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                      width=line_width, fill=line_color)
        im.save(img_path + '_act.jpg')
        quad_draw = ImageDraw.Draw(quad_im)
        txt_items = []
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):
            if np.amin(score) > 0:
                quad_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=2, fill='red')
                if cfg.predict_cut_text_line:
                    cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                                  img_path, s)
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        quad_im.save(img_path + '_predict.jpg')
        if cfg.predict_write2txt and len(txt_items) > 0:
            with open(img_path[:-4] + '.txt', 'w') as f_txt:
                f_txt.writelines(txt_items)

def predict_quad(model, img, pixel_threshold=cfg.pixel_threshold, quiet=False, img_name=None):
    """
    Args:
        model: 检测模型，要load_weights的
        img:  image 图片，文件类型
        pixel_threshold: 阈值
        quiet:
        img_name: 图片的名字

    Returns:
        text_recs_all：一个列表，每个元素是检测边界的quad值
        text_recs_len：text_recs_all的长度，一共检测到多少个区域
        img_all: 一个思维数组，img_all[0] 是img_to_array的结果

    """
    # 获取计算后的图像长宽
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    # 调整图像大小，便于预测
    img = img.resize((d_wight, d_height), Image.BILINEAR).convert('RGB')
    img = image.img_to_array(img)
    num_img = 1
    # 一个4维张量，也就是只有1个3维张量的4维张量
    img_all = np.zeros((num_img, d_height, d_wight, 3))
    img_all[0] = img

    # 将张量的数值大小调到【-1 1】
    img_ori = preprocess_input(img, mode='tf')  # suit tf tensor

    # 又整个一样的
    x = np.zeros((num_img, d_height, d_wight, 3))
    x[0] = img_ori

    # (sample, h, w, channels)
    y_pred = model.predict(x)

    text_recs_all = []
    text_recs_len = []
    for n in range(num_img):
        # (sample, rows, cols, 7_points_pred)
        y = y_pred[n]
        y[:, :, :3] = sigmoid(y[:, :, :3])
        cond = np.greater_equal(y[:, :, 0], pixel_threshold)
        activation_pixels = np.where(cond)  # fixme 返回元祖tuple类型 a[0]保存了纵坐标 a[1]保存横坐标
        quad_scores, quad_after_nms = nms(y, activation_pixels)

        text_recs = []
        x[n] = np.uint8(x[n])
        with image.array_to_img(img_all[n]) as im:     # Image.fromarray(x[n]) error ?
            im_array = x[n]

            # fixme 注意：拿去CRNN识别的是缩放后的图像
            scale_ratio_w = 1
            scale_ratio_h = 1

            quad_im = im.copy()
            draw = ImageDraw.Draw(im)
            # 拷贝一个原图像，在拷贝的图像上有文字的地方画线
            for i, j in zip(activation_pixels[0], activation_pixels[1]):
                px = (j + 0.5) * cfg.pixel_size
                py = (i + 0.5) * cfg.pixel_size
                line_width, line_color = 1, 'blue'
                if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                    if y[i, j, 2] < cfg.trunc_threshold:
                        line_width, line_color = 2, 'yellow'
                    elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                        line_width, line_color = 2, 'green'
                draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                           (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                           (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                           (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                           (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                          width=line_width, fill=line_color)

            if not img_name is None:
                im.save(root_temp + img_name + '_%d_.jpg' % n)

            quad_draw = ImageDraw.Draw(quad_im)
            for score, geo, s in zip(quad_scores, quad_after_nms,
                                     range(len(quad_scores))):
                if np.amin(score) > 0:
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='blue')

                    if cfg.predict_cut_text_line:
                        cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                                      img_name, s)

                    rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                    text_rec = np.reshape(rescaled_geo, (8,)).tolist()
                    text_recs.append(text_rec)
                elif not quiet:
                    print('quad invalid with vertex num less then 4.')

            if not img_name is None:
                quad_im.save(root_predict + img_name + '_%d_.jpg' % n)

        for t in range(len(text_recs)):
            text_recs_all.append(text_recs[t])

        text_recs_len.append(len(text_recs))

    return text_recs_all, text_recs_len, img_all

def predict_txt(east_detect, img_path, txt_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    scale_ratio_w = d_wight / img.width
    scale_ratio_h = d_height / img.height
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    txt_items = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    if cfg.predict_write2txt and len(txt_items) > 0:
        with open(txt_path, 'w') as f_txt:
            f_txt.writelines(txt_items)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='../data/003.jpg',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_path = args.path
    threshold = float(args.threshold)
    print(img_path, threshold)
    # img = image.load_img(img_path)
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)
    predict(east_detect, img_path, threshold)
    # text_recs_all, text_recs_len, img_all = predict_quad(east_detect, img, img_name='001')
    # print(text_recs_all)
    # print("-------------------------")
    # print(text_recs_len)
    # print("-------------------------")
    # print(img_all)
    # img = image.array_to_img(img_all[0])
    # img.show()
