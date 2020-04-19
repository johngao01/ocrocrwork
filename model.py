from dect.predict import predict, predict_quad
import argparse
from dect import cfg
from dect.network import East
from tensorflow.keras.preprocessing import image
import os
crnn_model_path = 'weights/crnn/netCRNN_4_48000.pth'
dect_weights_path = 'weights/east/east_model_weights_3T640.h5'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='data/003.jpg',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()


if __name__ == '__main__':
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(dect_weights_path)
    east_detect.summary()

    args = parse_args()
    img_path = args.path
    img = image.load_img(img_path).convert('RGB')

    threshold = args.threshold
    # img.show()
    im_name = img_path.split('/')[-1][:-4]
    print(im_name)
    text_recs_all, text_recs_len, img_all = predict_quad(east_detect, img, img_name=im_name)

    from reco.Net import net
    from reco import alphabets
    import torch

    alphabet = alphabets.alphabet
    nclass = len(alphabet) + 1
    model = net.CRNN(nclass)
    model.load_state_dict(torch.load(crnn_model_path))
    imgred = image.array_to_img(img_all[0])
    imgred.show()
    print("成功了")
    if len(text_recs_all) > 0:
        pass



