import torch

def calc_map(args, model, pos_m, pos_val_b):
    """ MAP """
    ui_mat = torch.mm(model.user_embd.weight, model.item_embd.weight.transpose(0, 1))
    ui_mat *= pos_m
    rank = ui_mat.argsort(1, descending=True)
    pos_k = pos_val_b.gather(1, rank)[:, :50].float()
    prec = pos_k.cumsum(1)
    prec_base = torch.ones_like(pos_k).cumsum(1)
    prec = prec / prec_base
    map_ = ((prec * (pos_k > 0)).sum(1) / ((pos_val_b > 0).sum(1) + 1e-8)).mean()
    return map_

def calc_recall(args, model, pos_m, pos_val_b):
    """ Recall """
    ui_mat = torch.mm(model.user_embd.weight, model.item_embd.weight.transpose(0, 1))
    ui_mat *= pos_m
    rank = ui_mat.argsort(1, descending=True)
    pos_k = pos_val_b.gather(1, rank)[:, :50].float()
    recall = (pos_k.sum(1) / pos_val_b.sum(1)).mean()
    return recall