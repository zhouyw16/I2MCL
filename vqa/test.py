import os
import time
import json
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import CLIPTokenizer, CLIPModel

import utils
from base_model import build
from dataset import Dictionary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--output', type=str, default='exp')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    return args


def _create_entry(img, question):
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question']}
    return entry


def _load_dataset(dataroot, img_id2val):
    question_path = os.path.join(dataroot, 'v2_OpenEnded_mscoco_test2015_questions.json')
    questions = sorted(json.load(open(question_path))['questions'], key=lambda x: x['question_id'])
    entries = []
    for question in questions:
        img_id = 'test2015/COCO_test2015_%012d.jpg' % (question['image_id'])
        entries.append(_create_entry(img_id2val[img_id], question))
    return entries


class VQATestDataset(Dataset):
    def __init__(self, dictionary, transform, dataroot='data'):
        super(VQATestDataset, self).__init__()

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.transform = transform

        t = time.time()
        imgroot = '/DATA/DATANAS1/zyw16/MMData/mscoco'
        self.img_id2idx = np.load(os.path.join(imgroot, 'test2015_img2id.npy'), allow_pickle=True).item()
        self.images = np.load(os.path.join(imgroot, 'test2015.npy'), allow_pickle=True)
        self.entries = _load_dataset(dataroot, self.img_id2idx)
        print('load images from npy file: %.2fs' % (time.time() - t))

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        tokenizer.padding_side = 'left'
        for entry in self.entries:
            clip_tokens = tokenizer.encode(entry['question'], padding='max_length', truncation=True, max_length=max_length)
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(clip_tokens), max_length)
            utils.assert_eq(len(tokens), max_length)
            entry['clip_token'] = clip_tokens
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            clip_question = torch.from_numpy(np.array(entry['clip_token']))
            entry['clip_token'] = clip_question
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

    def __getitem__(self, index):
        entry = self.entries[index]
        image = self.transform(Image.fromarray(self.images[entry['image']].astype('uint8')).convert('RGB'))
        clip_question = entry['clip_token']
        question = entry['q_token']
        question_id = entry['question_id']
        return image, question, clip_question, question_id

    def __len__(self):
        return len(self.entries)


def test(model, dataloader, output):
    label2ans = dataloader.dataset.label2ans
    question_ids, answers = [], []
    model.eval()
    with torch.no_grad():
        for v, q, clip_q, i in tqdm(dataloader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            clip_q = Variable(clip_q).cuda()
            pred = model(v, q)
            ans = pred.argmax(1).tolist()
            question_ids += i.tolist()
            answers += [label2ans[a] for a in ans]

    json_data = [{'question_id': question_id, 'answer': answer} \
                for question_id, answer in zip(question_ids, answers)]
    json_file = os.path.join(output, 'test.json')
    with open(json_file, 'w') as writer: json.dump(json_data, writer)


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
    trans = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(MEAN, STD)])
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    test_dset = VQATestDataset(dictionary, trans)
    batch_size = args.batch_size
    print('load data: %.2fs' % (time.time() - t))

    t = time.time()
    model = build(test_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    print('build model: %.2fs' % (time.time() - t))

    test_loader = DataLoader(test_dset, batch_size, shuffle=False, pin_memory=True, num_workers=4)
    model_path = os.path.join(args.output, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    test(model, test_loader, args.output)
