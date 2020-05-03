from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon
from ui_ocr import Ui_ui_ocr
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from dect.predict import predict_quad
from dect.network import East
from tensorflow.keras.preprocessing import image
crnn_model_path = 'weights/crnn/netCRNN_4_48000.pth'
dect_weights_path = 'weights/east/east_model_weights_3T640.h5'
running_mode = 'gpu'


def dectAndReco(filepath):
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(dect_weights_path)
    img = image.load_img(filepath).convert('RGB')
    im_name = filepath.split('/')[-1][:-4]
    text_recs_all, text_recs_len, img_all = predict_quad(
        east_detect, img, img_name=im_name)
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
        crnn_model.load_state_dict(
            torch.load(
                crnn_model_path,
                map_location='cpu'))

    if len(text_recs_all) > 0:
        texts = predict_text(
            crnn_model,
            text_recs_all,
            text_recs_len,
            img_all,
            img_name=im_name)

    return texts


# 主窗口
class Main_window(QWidget, Ui_ui_ocr):
    def __init__(self):
        super(Main_window, self).__init__()
        self.imgpath = ''
        self.setupUi(self)

        # 逻辑处理
        self.initUI()

    # 后台的逻辑处理
    def initUI(self):
        self.open.clicked.connect(self.loadimg)
        self.reco.clicked.connect(self.beginreco)
        self.clean.clicked.connect(self.cleanimg)
        self.copyget.clicked.connect(self.copyresult)
        self.ocr.clicked.connect(self.changetoocr)
        self.his.clicked.connect(self.changetohis)

    def changetoocr(self):
        # 调到ocr识别界面
        self.stackedWidget.setCurrentIndex(1)

    def changetohis(self):
        # 调到历史识别界面
        self.stackedWidget.setCurrentIndex(2)

    def loadimg(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, '打开文件', './data', '图像文件(*.jpg *.png)')
        self.imgpath = fname
        self.imgsrc = QPixmap(fname).scaled(
            self.img_show.width(), self.img_show.height())
        self.img_show.setPixmap(self.imgsrc)

    def beginreco(self):
        temp = ''
        if self.imgpath == temp:
            print(self.imgpath)
        else:
            txt = dectAndReco(self.imgpath)
            print(txt)
            self.result_.setText(txt)

    def cleanimg(self):
        self.imgpath = ''
        self.imgsrc = None
        self.img_show.clear()

    def copyresult(self):
        temp = ''
        if self.result_.text() == temp:
            QMessageBox.information(
                self, "提示", "还没有结果呢", QMessageBox.Yes)

        else:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.result_.text())
