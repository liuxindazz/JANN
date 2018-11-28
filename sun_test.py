from __future__ import print_function
import os
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch
from torch.autograd import Variable
from torchvision import transforms
from dataset.sun_data_loader import GetLoader
from torchvision import datasets
import visdom

vis = visdom.Visdom(env='dann_sun_rgbd')
def test(dataset_name, epoch):
    assert dataset_name in ['rgb', 'hha']

    model_root = os.path.join( 'models')
    image_root = os.path.join( '/1116', 'SUN', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 224
    alpha = 0

    """load data"""

    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_list = os.path.join('/1116', 'SUN', 'test_label.txt')
    dataset = GetLoader(
        data_root=os.path.join(image_root, 'test'),
        data_list=test_list,
        transform=img_transform
    )


    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'sun_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        class_output, _ = my_net(input_data=inputv_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        tmp = pred.eq(classv_label.data.view_as(pred)).cpu()
        k = torch.sum(tmp).item()
        n_correct += k
        n_total += batch_size

        i += 1

    accu = n_correct * 1.0 / n_total

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
    vis.line(X=torch.FloatTensor([epoch+1]), 
                 Y=torch.FloatTensor([accu]), 
                 win=dataset_name, 
                 opts=dict(title=dataset_name),
                 update='append' if i> 0 else None)