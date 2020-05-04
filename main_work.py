import pymysql
import time
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from ui_ocr import Ui_ui_ocr
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from dect.predict import predict_quad
from dect.network import East
from tensorflow.keras.preprocessing import image
crnn_model_path = 'weights/crnn/netCRNN_4_48000.pth'
dect_weights_path = 'weights/east/east_model_weights_3T640.h5'
running_mode = 'gpu'


def init_hispage(self):
    # 初始化历史识别界面
    self.instructions = QtWidgets.QLabel(self.his_page)
    self.instructions.setGeometry(QtCore.QRect(230, 30, 521, 91))
    font = QtGui.QFont()
    font.setPointSize(20)
    self.instructions.setFont(font)
    self.instructions.setAlignment(QtCore.Qt.AlignCenter)
    self.instructions.setObjectName("instructions")
    self.instructions.setText("ocr识别记录")

    db = pymysql.connect(
        "localhost",
        "root",
        "123456",
        "ocrhis",
        charset='utf8')
    # 获取游标、数据
    cur = db.cursor()
    cur.execute("SELECT * FROM his")
    data = cur.fetchall()
    # 数据列名
    col_lst = ['序号', '文件路径', '识别时间', '识别结果']

    # 数据的大小
    row = len(data)  # 行
    if row > 0:
        vol = len(data[0])
    else:
        row = 0
        vol = 4

    # 插入表格
    self.histable = QTableWidget(row, vol, self.his_page)
    self.histable.setGeometry(QtCore.QRect(100, 130, 821, 521))
    font = QtGui.QFont('微软雅黑', 10)

    # 设置字体、表头
    self.histable.horizontalHeader().setFont(font)
    self.histable.setHorizontalHeaderLabels(col_lst)
    # 设置竖直方向表头不可见
    self.histable.verticalHeader().setVisible(False)
    self.histable.setFrameShape(QFrame.Box)
    # 构建表格插入数据
    if row >= 1:
        for i in range(row):
            for j in range(vol):
                temp_data = data[i][j]  # 临时记录，不能直接插入表格
                data1 = QTableWidgetItem(str(temp_data))  # 转换后可插入表格
                self.histable.setItem(i, j, data1)
    db.close()

    self.histable.horizontalHeader().setSectionResizeMode(
        0, QHeaderView.ResizeToContents)
    self.histable.horizontalHeader().setSectionResizeMode(
        2, QHeaderView.ResizeToContents)
    self.histable.horizontalHeader().setSectionResizeMode(
        1, QHeaderView.ResizeToContents)
    self.histable.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)


def addhis(path, txt):
    db = pymysql.connect(
        "localhost",
        "root",
        "123456",
        "ocrhis",
        charset='utf8')
    cursor = db.cursor()
    currenttime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        sql = "insert into his(filepath,recotime,result) values ('%s', '%s', '%s') " % (
            path, currenttime, txt)
        cursor.execute(sql)
        db.commit()
    except BaseException as e:
        print(e)


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
        self.statusBar.showMessage("欢迎使用此OCR")

    def changetohis(self):
        init_hispage(self)
        self.stackedWidget.setCurrentIndex(2)
        self.statusBar.showMessage("历史记录加载完成")

    def loadimg(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, '打开文件', './data', '图像文件(*.jpg *.png)')
        self.imgpath = fname
        self.imgsrc = QPixmap(fname).scaled(
            self.img_show.width(), self.img_show.height())
        self.img_show.setPixmap(self.imgsrc)
        self.statusBar.showMessage("上传图片成功，图片路径："+fname)

    def beginreco(self):
        temp = ''
        if self.imgpath == temp:
            print(self.imgpath)
        else:
            txt = dectAndReco(self.imgpath)
            self.result_.setText(txt)
            addhis(self.imgpath, txt)
            self.statusBar.showMessage("识别成功和添加记录成功")


    def cleanimg(self):
        self.imgpath = ''
        self.imgsrc = None
        self.img_show.clear()
        self.statusBar.showMessage("清除图片成功")

    def copyresult(self):
        temp = ''
        if self.result_.text() == temp:
            self.statusBar.showMessage("无识别结果")

        else:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.result_.text())
            self.statusBar.showMessage("复制成功")
