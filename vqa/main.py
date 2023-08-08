import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

import utils
from base_model import build
from train import train, evaluate
from dataset import Dictionary, VQAImageTextDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
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
    MEAN = [0.47087017, 0.44870174, 0.41087511]
    STD = [0.27794643, 0.27372972, 0.28868105]
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
    dictionary = Dictionary.load_from_file('data/vqa-v2/dictionary.pkl')
    train_dset = VQAImageTextDataset('train', dictionary, train_trans)
    eval_dset = VQAImageTextDataset('val', dictionary, eval_trans)
    batch_size = args.batch_size
    print('load data: %.2fs' % (time.time() - t))

    t = time.time()
    model = build(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/vqa-v2/glove6b_init_300d.npy')
    print('build model: %.2fs' % (time.time() - t))

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, pin_memory=True, num_workers=4)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, pin_memory=True, num_workers=4)
    if args.mode == 'train': train(model, train_loader, eval_loader, args.epochs, args.output)

    model_path = os.path.join(args.output, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    eval_score, bound = evaluate(model, eval_loader)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
