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
    #grid_size = args.grid_size
    #pair_num = args.pair_num
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

def compute_order_interaction_img(args, model, img_adv, lbl, point_list, players, r):
    #device = args.device if args.device == "cpu" else int(args.device)
    device = 0  # Do not modify
    model.to(device)

    with torch.no_grad():
        model.eval()
        img_size = args.img_size
        channels = img_adv.size(0)
        #print("channels:", channels)
        tic = time.time()
        adv_interaction = []  # save order interactions of point-pairs in adv image
        #print("players.shape:", np.shape(players))  #(200, 100, 241)

        forward_mask = []
        #print("point_list:", np.shape(point_list))  # (200, 2)
        for p, pt in enumerate(point_list):
            point1, point2 = pt[0], pt[1]
            # print("i: %d, j: %d" % (point1, point2))

            players_thispair = players[p]
            #print("players_thispair.shape:", np.shape(players[p]))   #(100, 241)
            m = int((args.grid_size ** 2 - 2) * r)  # m-order
            mask = torch.zeros((4 * args.sample_num, channels, args.grid_size * args.grid_size)).to(device)

            for k in range(args.sample_num):
                mask[4*k:4*(k+1), :, players_thispair[k]] = 1  # S
                mask[4*k+1, :, point1] = 1  # S U {i}
                mask[4*k+2, :, point2] = 1  # S U {j}
                mask[4*k, :, point1] = 1
                mask[4*k, :, point2] = 1    # S U {i,j}
            mask = mask.view(4 * args.sample_num, channels, args.grid_size, args.grid_size)
            mask = F.interpolate(mask.clone(), size=[img_size, img_size], mode="nearest").float()
            #print("mask.shape:", np.shape(mask))  # torch.Size([400, 3, 32, 32])

            forward_mask.append(mask)
            if (len(forward_mask) < args.cal_batch // args.sample_num) and (p < args.pair_num - 1):
                continue
            else:
                forward_batch = len(forward_mask) * args.sample_num
                batch_mask = torch.cat(forward_mask, dim=0)
                expand_img_adv = img_adv.expand(4 * forward_batch, -1, img_size, img_size).clone()
                masked_img_adv = batch_mask * expand_img_adv
                #print("masked_img_adv.shape:", masked_img_adv.size())     # torch.Size([400, 3, 32, 32])
                #print("masked_img_adv:", masked_img_adv)
                print("batch_mask.shape:",batch_mask.size())
                print("batch_mask:", batch_mask)
                # print("batch_mask.shape:",batch_mask.size())            # torch.Size([400, 3, 32, 32])
                # print("expand_img_adv.shape", np.shape(expand_img_adv))         # torch.Size([400, 3, 32, 32])

                output_adv = model(masked_img_adv)             # torch.Size([400, 10])
                #print("output_adv.shape:", output_adv.size())
                #print("output_adv:", output_adv)

                if args.softmax_type == 'normal':
                    y_adv = F.log_softmax(output_adv, dim=1)[:, lbl]  # lbl[0]
                    #print("y_adv.shape:", y_adv.size())        # torch.Size([400])
                    #print("y_adv:", y_adv)                     # 都在-2.33 到 -2.30之间

                for k in range(forward_batch):
                    score_adv = y_adv[4 * k] + y_adv[4 * k + 3] - y_adv[4 * k + 1] - y_adv[4 * k + 2]
                    #print("score_adv:", score_adv)             # 一个值，0.0001 到 0.001 这个数量级左右
                    adv_interaction.append(score_adv.item())
                forward_mask = []

        #print("adv_interaction.shape:", np.shape(adv_interaction))  # (20000,)
        #print("adv_interaction:",adv_interaction )
        adv_interaction = np.array(adv_interaction).reshape(-1, args.sample_num)

        print('Image: 1 张图 , time: %.3f' % ( time.time() - tic))   # time: 15.983s
        print('--------------------------')
        '''
        tmp = "img{}_interaction.npy".format(lbl)
        #np.save(os.path.join(args.m_inter_path, tmp), ori_interaction)  # (pair_num, sample_num)
        np.save(os.path.join(args.m_inter_path, "adv_" + tmp), adv_interaction)
        '''
        #print("adv_interaction.shape:", np.shape(adv_interaction))     # (200, 100)
        print("adv_interaction:",adv_interaction )

        return adv_interaction

def gen_mask(r, pair_num, sample_num, grid_size, img_size, local_size=1):
    point_list = gene_local_points(grid_size, pair_num, local_size)       # (pair_num: 200, 2)
    #print(f"point_list: {point_list}, point_list.shape:{ np.shape(point_list) }")
    players = gen_context(point_list, grid_size, sample_num, r=r)         # (pair_num: 200, context_num: 100, r: 58)
    #print(f"players: {players}, players.shape:{ np.shape(players) }")
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
    print(f"batch_mask: {batch_mask}, batch_mask.shape:{ batch_mask.size() }")  # [80000, 3, 32, 32]
    return batch_mask


def compute_order_interaction_img_timecut(net, imgs, lbls, mask, pair_num, sample_num):
    mask_size = mask.size(0)    # pair_num*sample_num*4 : 80000
    assert pair_num * sample_num * 4 == mask_size
    N = imgs.size(0)
    print(f"N:{N}")
    expand_imgs = imgs.repeat_interleave(mask_size, dim=0)
    batch_mask = mask.repeat(N, 1, 1, 1)
    masked_imgs = batch_mask * expand_imgs
    print(f"masked_imgs: {masked_imgs}, masked_imgs.shape:{ masked_imgs.size() }")

    net.eval()
    logits = F.log_softmax(net(masked_imgs), dim=-1)  # Shape: (N*pair_num*sample_num*4, num_classes)
    assert logits.shape[0] == mask_size * N

    lbls_mask = torch.repeat_interleave(lbls, repeats=mask_size, dim=0).reshape(mask_size * N, 1)
    lbl_logits = torch.gather(logits, dim=1, index=lbls_mask).squeeze(dim=1)
    lbl_logits = lbl_logits.reshape(N * pair_num, sample_num * 4)
    inter = lbl_logits[:, 0::4] + lbl_logits[:, 3::4] - lbl_logits[:, 1::4] - lbl_logits[:, 2::4]

    inter = inter.reshape(N, pair_num, sample_num)  # shape: (N, pair_num, sample_num)
    # inter = inter.mean(dim=2)
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
    # 传入一个batch图像，传出一个batch图像的interaction , 先仅计算单阶 0.95n
    #delta_f_batch = 0

    for r in args.ratios:
        mask = gen_mask(r, args.pair_num, args.sample_num, args.grid_size, args.img_size, local_size=1)
        delta_f_batch = compute_order_interaction_img_timecut(model, image, label, mask, args.pair_num, args.sample_num) # shape: (N, pair_num, sample_num)

        inter_m_mean = torch.mean(delta_f_batch, dim=2)
        inter_m_mean = torch.mean(inter_m_mean, dim=1)

    return inter_m_mean