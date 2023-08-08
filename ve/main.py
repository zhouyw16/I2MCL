import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

import utils
from base_model import build
from train import train, evaluate
from dataset import Dictionary, VEImageTextDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--output', type=str, default='exp')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.output = 'saved_models/' + args.output
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    t = time.time()
    MEAN = [0.44430269, 0.42129134, 0.38488099]
    STD = [0.28511056, 0.27731498, 0.28582974]
    train_trans = T.Compose([
        T.RandomResizedCrop([224, 224], scale=(0.5, 1.0), 
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD)])
    eval_trans = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(MEAN, STD)])
    dictionary = Dictionary.load_from_file('data/snli-ve/dictionary.pkl')
    img_id2idx = np.load('data/flickr30k/img2id.npy', allow_pickle=True).item()
    images = np.load('data/flickr30k/img.npy', allow_pickle=True)
    print('load images from npy file: %.2fs' % (time.time() - t))

    train_dset = VEImageTextDataset('train', dictionary, images, img_id2idx, train_trans)
    dev_dset = VEImageTextDataset('dev', dictionary, images, img_id2idx, eval_trans)
    test_dset = VEImageTextDataset('test', dictionary, images, img_id2idx, eval_trans)
    batch_size = args.batch_size
    print('load data: %.2fs' % (time.time() - t))

    t = time.time()
    model = build(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/snli-ve/glove6b_init_300d.npy')
    print('build model: %.2fs' % (time.time() - t))

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, pin_memory=True, num_workers=4)
    dev_loader = DataLoader(dev_dset, batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dset, batch_size, shuffle=False, pin_memory=True, num_workers=4)
    if args.mode == 'train': train(model, train_loader, dev_loader, test_loader, args.epochs, args.output)

    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    model_path = os.path.join(args.output, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    dev_score = evaluate(model, dev_loader)
    test_score = evaluate(model, test_loader)
    logger.write('\tdev score: %.2f' % (100 * dev_score))
    logger.write('\ttest score: %.2f' % (100 * test_score))
