import Config
import lib.dataset
import lib.convert
import lib.utility
import torch
import Net.net as Net
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CTCLoss

# this script test a model accuracy on test_lmdb

image = torch.FloatTensor(
        Config.batch_size,
        3,
        Config.img_height,
        Config.img_width)
text = torch.IntTensor(Config.batch_size * 5)
length = torch.IntTensor(Config.batch_size)


def val(net, da, criterion, max_iter=100):
    print('Start val')
    net.eval()
    with torch.no_grad():
        data_loader = torch.utils.data.DataLoader(
            da, shuffle=True, batch_size=Config.batch_size, num_workers=int(Config.data_worker))
        val_iter = iter(data_loader)

        i = 0
        n_correct = 0
        loss_avg = lib.utility.averager()

        max_iter = min(max_iter, len(data_loader))
        for i in range(max_iter):
            data = val_iter.next()
            i += 1
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            lib.dataset.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)
            lib.dataset.loadData(text, t)
            lib.dataset.loadData(length, l)

            preds = net(image)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            cost = criterion(preds, text, preds_size, length) / batch_size
            loss_avg.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            list_1 = []
            for i in cpu_texts:
                list_1.append(i.decode('utf-8', 'strict'))
            for pred, target in zip(sim_preds, list_1):
                if pred == target:
                    n_correct += 1

    accuracy = n_correct / float(max_iter * Config.batch_size)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    return loss_avg.val(), accuracy


test_dataset = lib.dataset.lmdbDataset(
        root=Config.test_data, transform=lib.dataset.resizeNormalize(
            (Config.img_width, Config.img_height)))

converter = lib.convert.strLabelConverter(Config.alphabet)
n_class = len(Config.alphabet) + 1

net = Net.CRNN(n_class)
checkpoint = torch.load('checkpoint1.pt.tar')
#  引用参数
net.load_state_dict(checkpoint['model_state_dict'])
criterion = CTCLoss()

loss, accuracy = val(net, test_dataset, criterion)


