import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import CLIPModel

import utils
from superloss import Superloss
from mgda import gradient_weights


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').sum(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    global_step = 1
    teacher = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').cuda()
    sp, v_sp, q_sp = Superloss(), Superloss(), Superloss()

    for epoch in range(num_epochs):
        total_loss = 0
        v_total_loss = 0
        q_total_loss = 0
        train_score = 0
        pre_time = 0
        t = time.time()

        model.train()
        for v, q, clip_q, a in tqdm(train_loader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred, v_emb, q_emb = model(v, q, return_emb=True)
            loss = instance_bce_with_logits(pred, a)

            pre_t = time.time()
            with torch.no_grad():
                clip_q = Variable(clip_q).cuda()
                v_feat = teacher.get_image_features(v)
                q_feat = teacher.get_text_features(clip_q, attention_mask=(clip_q < 49407))
            pre_time += time.time() - pre_t

            v_loss = F.mse_loss(v_emb, v_feat, reduction='none').sum(1)
            q_loss = F.mse_loss(q_emb, q_feat, reduction='none').sum(1)

            uni_v_grad = torch.autograd.grad(v_loss.mean(), v_emb, retain_graph=True)
            uni_q_grad = torch.autograd.grad(q_loss.mean(), q_emb, retain_graph=True)
            mul_v_grad = torch.autograd.grad(loss.mean(), v_emb, retain_graph=True)
            mul_q_grad = torch.autograd.grad(loss.mean(), q_emb, retain_graph=True)

            v_weight = gradient_weights(uni_v_grad, mul_v_grad)
            q_weight = gradient_weights(uni_q_grad, mul_q_grad)
            global_step += 1
            
            if v_weight < q_weight: 
                mul_q_loss = instance_bce_with_logits(model.logits(v_emb.detach(), q_emb), a)
                (v_sp(v_loss) + sp(mul_q_loss)).backward()
            else:
                mul_v_loss = instance_bce_with_logits(model.logits(v_emb, q_emb.detach()), a)
                (sp(mul_v_loss) + q_sp(q_loss)).backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum().item()
            total_loss += loss.mean().item() * v.size(0)
            v_total_loss += v_loss.mean().item() * v.size(0)
            q_total_loss += q_loss.mean().item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        v_total_loss /= len(train_loader.dataset)
        q_total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        eval_score, bound = evaluate(model, eval_loader)

        logger.write('epoch %d, time: %.2fs' % (epoch, time.time() - t - pre_time))
        logger.write('\ttrain_loss: %.2f, v_loss: %.2f, q_loss: %.2f, score: %.2f' 
                    % (total_loss, v_total_loss, q_total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    model.eval()
    with torch.no_grad():
        for v, q, clip_q, a in tqdm(dataloader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            pred = model(v, q)
            batch_score = compute_score_with_logits(pred, a.data).sum().item()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum().item()
            num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
