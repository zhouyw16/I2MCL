import numpy as np
import torch


def gradient_weights(grad1, grad2):
    v1v1, v1v2, v2v2 = 0.0, 0.0, 0.0
    for g1, g2 in zip(grad1, grad2):
        v1v1 += torch.mul(g1, g1).sum().item()
        v1v2 += torch.mul(g1, g2).sum().item()
        v2v2 += torch.mul(g2, g2).sum().item()

    if v1v2 >= v2v2:
        # Case: Fig 1, first column
        gamma = 0.001
        cost = v2v2
    elif v1v2 >= v1v1:
        # Case: Fig 1, third column
        gamma = 0.999
        cost = v1v1
    else:
        # Case: Fig 1, second column
        gamma = (v2v2 - v1v2) / max(v1v1 + v2v2 - 2.0 * v1v2, 1e-8)
        cost = v2v2 + gamma * (v1v2 - v2v2)
    try:
        assert cost < 1e12
    except:
        print(gamma, cost, v1v1, v1v2, v2v2)
        exit()
    return gamma


def gradient_norm(grad, loss, type):
    if type == 'l2':
        gn = np.sqrt(np.sum([g.pow(2).sum().item() for g in grad]))
    elif type == 'loss':
        gn = loss
    elif type == 'loss+':
        gn = loss * np.sqrt(np.sum([g.pow(2).sum().item() for g in grad]))
    elif type == 'none':
        gn = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return tuple(g / gn for g in grad)
