import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import time
import numpy as np

device = 0

def gene_local_points(grid_size, pair_num, local_size=1):
    neighbor = []
    for j in range(-local_size, local_size + 1):
        for h in range(-local_size, local_size + 1):
            if j == 0 and h == 0:
                pass
            else:
                item = [j, h]
                neighbor.append(item)

    neighbor = np.array(neighbor)

    tot_pairs = []
    for k in range(pair_num):
        while True:
            x1 = np.random.randint(0, grid_size)
            y1 = np.random.randint(0, grid_size)
            point1 = x1 * grid_size + y1

            np.random.shuffle(neighbor)
            x2 = x1 + neighbor[0][0]
            y2 = y1 + neighbor[0][1]
            if x2 < 0:
                x2 = 0
            elif x2 >= grid_size:
                x2 = grid_size - 1
            if y2 < 0:
                y2 = 0
            elif y2 >= grid_size:
                y2 = grid_size - 1
            point2 = x2 * grid_size + y2

            if point1 == point2:  # bugFix
                continue

            if [point1, point2] in tot_pairs or [point2, point1] in tot_pairs:
                continue
            else:
                break
        tot_pairs.append(list([point1, point2]))

    tot_pairs = np.array(tot_pairs)    # 这里是否需要np呢
    return tot_pairs

def gen_context(point_list, grid_size, sample_num, r=0.95):
    players = []
    m = int((grid_size ** 2 - 2) * r)  # m-order
    for p, (point1, point2) in enumerate(point_list):
        # m-order interactions
        context = list(range(grid_size ** 2))
        context.remove(point1)
        context.remove(point2)

        players_thispair = []
        for k in range(sample_num):
            players_thispair.append(np.random.choice(context, m, replace=False))

        players.append(players_thispair)

    players = np.array(players)
    return players


def gen_mask(r, pair_num, sample_num, grid_size, img_size, local_size=1):
    point_list = gene_local_points(grid_size, pair_num, local_size)       # (pair_num: 200, 2)
    players = gen_context(point_list, grid_size, sample_num, r=r)         # (pair_num: 200, context_num: 100, r: 58)
    channels = 3
    batch_mask = torch.zeros((pair_num*sample_num*4, grid_size**2), dtype=torch.uint8, device=device)
    for idx, ((point1, point2), players_thispair) in enumerate(zip(point_list, players)):
        mask = torch.zeros((sample_num, grid_size**2), dtype=torch.uint8, device=device).to(device)    # (sample_num, grid_size**2)
        mask = mask.scatter_(dim=1, index=torch.tensor(players_thispair, device=device), value=1)    # (sample_num, grid_size**2)
        mask = torch.repeat_interleave(mask, repeats=4, dim=0)     # (4 * sample_num, channels, grid_size**2)
        mask[0::4, [point1, point2]] = 1    # S U {i,j}
        mask[1::4, point1] = 1  # S U {i}
        mask[2::4, point2] = 1  # S U {j}
        batch_mask[4*sample_num*idx:4*sample_num*(idx+1)] = mask

    batch_mask = torch.unsqueeze(batch_mask, dim=1).expand(-1, channels, -1)    # (pair_num*sample_num*4, channels, grid_size**2)
    batch_mask = batch_mask.view(pair_num*sample_num*4, channels, grid_size, grid_size)
    batch_mask = F.interpolate(batch_mask.clone(), size=[img_size, img_size], mode="nearest")
    return batch_mask


def compute_order_interaction_img(net, imgs, lbls, mask, pair_num, sample_num):
    mask_size = mask.size(0)    # pair_num*sample_num*4 : 80000
    assert pair_num * sample_num * 4 == mask_size
    N = imgs.size(0)
    expand_imgs = imgs.repeat_interleave(mask_size, dim=0)
    batch_mask = mask.repeat(N, 1, 1, 1)
    masked_imgs = batch_mask * expand_imgs

    net.eval()
    logits = F.log_softmax(net(masked_imgs), dim=-1)  # Shape: (N*pair_num*sample_num*4, num_classes)
    assert logits.shape[0] == mask_size * N

    lbls_mask = torch.repeat_interleave(lbls, repeats=mask_size, dim=0).reshape(mask_size * N, 1)
    lbl_logits = torch.gather(logits, dim=1, index=lbls_mask).squeeze(dim=1)
    lbl_logits = lbl_logits.reshape(N * pair_num, sample_num * 4)
    inter = lbl_logits[:, 0::4] + lbl_logits[:, 3::4] - lbl_logits[:, 1::4] - lbl_logits[:, 2::4]

    inter = inter.reshape(N, pair_num*sample_num)  # shape: (N, pair_num, sample_num)
    # inter = inter.mean(dim=2)
    return inter



def images_mask(N, img_size, r, grid_size, pair_num, sample_num):
    m = int((grid_size ** 2 - 2) * r)    # number of super pixel equal to 1
    mask_size = 4*pair_num*sample_num

    tmp = np.tile(np.arange(grid_size*grid_size), (N*pair_num, 1))
    tmp = np.apply_along_axis(np.random.permutation, axis=1, arr=tmp)
    tmp = np.repeat(tmp, axis=0, repeats=sample_num)
    point_pairs, contexts = tmp[:, :2], tmp[:, 2:]
    contexts = np.apply_along_axis(np.random.choice, axis=1, arr=contexts, size=m, replace=False)
    del tmp
    
    mask = np.zeros((N*pair_num*sample_num*4, grid_size*grid_size), dtype=np.uint8)
    np.put_along_axis(mask, np.repeat(contexts, axis=0, repeats=4), axis=1, values=1)    # S
    np.put_along_axis(mask[0::4], point_pairs, axis=1, values=1)    # S U {i,j}
    np.put_along_axis(mask[1::4], point_pairs[:, [0]], axis=1, values=1)    # S U {i}
    np.put_along_axis(mask[2::4], point_pairs[:, [1]], axis=1, values=1)    # S U {j}
    
    assert mask.shape[0] == 4*pair_num*sample_num*N

    mask = mask.reshape(4*pair_num*sample_num*N, 8, 8)

    return mask

def masked(images, r, grid_size, pair_num, sample_num):
    N, channels, img_size = images.size(0), images.size(1), images.size(2)
    mask_size = 4*pair_num*sample_num
    masks = images_mask(N, img_size, r, grid_size, pair_num, sample_num)
    expand_images = images.repeat_interleave(mask_size, dim=0)
    size = img_size//grid_size

    masks = torch.from_numpy(masks).to(images.device)
    masks = torch.unsqueeze(masks, dim=1).expand(-1, channels, -1, -1)
    pooling_images = (expand_images.reshape(N*mask_size, 3, grid_size, size, grid_size, size).sum(axis=(-3, -1)))*(1-masks)/(size**2)
    
    masks = F.interpolate(masks, size=[img_size, img_size], mode="nearest")
    pooling_images = F.interpolate(pooling_images, size=[img_size, img_size], mode="nearest")
    
    masked_images = masks*expand_images + pooling_images
    
    return masked_images


def compute_order_interaction_img_new(net, imgs, lbls, args):
    masked_imgs = masked(imgs, args.ratios[0], args.grid_size, args.pair_num, args.sample_num)
    mask_size = 4*args.pair_num*args.sample_num
    N = imgs.size(0)

    net.eval()
    logits = F.log_softmax(net(masked_imgs), dim=-1)  # Shape: (N*pair_num*sample_num*4, num_classes)
    assert logits.shape[0] == mask_size * N

    lbls_mask = torch.repeat_interleave(lbls, repeats=mask_size, dim=0).reshape(mask_size * N, 1)
    lbl_logits = torch.gather(logits, dim=1, index=lbls_mask).squeeze(dim=1)
    lbl_logits = lbl_logits.reshape(N * args.pair_num, args.sample_num * 4)
    inter = lbl_logits[:, 0::4] + lbl_logits[:, 3::4] - lbl_logits[:, 1::4] - lbl_logits[:, 2::4]

    inter = inter.reshape(N, args.pair_num*args.sample_num)  # shape: (N, pair_num, sample_num)
    return inter



def get_interaction_all(args, all_imgs, delta_f , norm_factor_adv=1):
    adv_orders_inter_mean = []
    adv_orders_inter_std = []
    for r in args.ratios:
        coef = 1
        adv_r_inters = []     # interactions for all imgs of this order
        for t in all_imgs:
            adv_inter = delta_f    # np.load(os.path.join(inters_dir, "adv_" + tmp))
            adv_inter = np.mean(adv_inter, axis=1)  # pair_num
            adv_inter = coef * adv_inter
            adv_inter = adv_inter / norm_factor_adv
            adv_r_inters.append(adv_inter)

        adv_r_inters = np.array(adv_r_inters) # (imgs_num, pair_num)

        adv_orders_inter_mean.append(np.mean(adv_r_inters))
        adv_orders_inter_std.append(np.std(adv_r_inters, ddof=1))

    return np.array(adv_orders_inter_mean), np.array(adv_orders_inter_std)

def inter_m_order(args, model, image, label, logger):
    r = args.ratios[0]
    mask = gen_mask(r, args.pair_num, args.sample_num, args.grid_size, args.img_size, local_size=1)
    # delta_f_batch = compute_order_interaction_img(model, image, label, mask, args.pair_num, args.sample_num) # shape: (N, pair_num, sample_num)
    delta_f_batch = compute_order_interaction_img_new(model, image, label, args)

    # inter_m_mean = torch.mean(delta_f_batch, dim=2)
    # inter_m_mean = torch.mean(inter_m_mean, dim=1)

    return delta_f_batch
