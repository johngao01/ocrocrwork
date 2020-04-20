from dect.predict import predict, predict_quad
import argparse
from dect import cfg
from dect.network import East
from tensorflow.keras.preprocessing import image
crnn_model_path = 'weights/crnn/netCRNN_4_48000.pth'
dect_weights_path = 'weights/east/east_model_weights_3T640.h5'
running_mode = 'gpu'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='data/004.png',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()


if __name__ == '__main__':
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(dect_weights_path)
    # east_detect.summary()

    args = parse_args()
    img_path = args.path
    img = image.load_img(img_path).convert('RGB')

    threshold = args.threshold
    # img.show()
    im_name = img_path.split('/')[-1][:-4]
    # print(im_name)
    text_recs_all, text_recs_len, img_all = predict_quad(east_detect, img, img_name=im_name)
    # print("检测成功了")
    # print(img_all[0].shape)
    # imgred = image.array_to_img(img_all[0])
    # imgred.show()

    # 这样导入就不会发生同时 import tensorflow 和 pytorch 的 cudnn 错误
    from reco.Net import net
    from reco import alphabets
    import torch
    from reco.infer import predict_text

    alphabet = alphabets.alphabet
    nclass = len(alphabet) + 1
    crnn_model = net.CRNN(nclass)
    if running_mode == 'gpu' and torch.cuda.is_available():
        model = crnn_model.cuda()
        model.load_state_dict(torch.load(crnn_model_path))
    else:
        crnn_model.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))

    if len(text_recs_all) > 0:
        texts = predict_text(crnn_model, text_recs_all, text_recs_len, img_all, img_name=im_name)

    print("done")


