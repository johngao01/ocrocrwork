import alphabets

raw_folder = ''
train_data = 'train_lmdb/'
test_data = 'test_lmdb/'
random_sample = True
random_seed = 1111
using_cuda = True
keep_ratio = False
gpu_id = '0'
model_dir = './pretrain_train'
# if data_worker > 0,some error raise on windows, why?
data_worker = 0
batch_size = 64
img_height = 32
img_width = 168
alphabet = alphabets.alphabet
epoch = 20
# my train dataset has 300,000 samples
log_interval = 500
display_interval = 20
save_interval = 1000
test_interval = 400
test_disp = 20
test_batch_num = 32
lr = 0.0001
beta1 = 0.5
infer_img_w = 160
