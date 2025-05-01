import time
from options.train_options import TrainOptions
from models.networks import VGGLoss, save_checkpoint
from models.afwm import TVLoss, AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import cv2
import datetime


def CreateDataset(opt):
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


if __name__ == "__main__":

    opt = TrainOptions().parse()

    path = 'runs/' + opt.name
    os.makedirs(path, exist_ok=True)

    os.makedirs('sample', exist_ok=True)
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')


    start_epoch, epoch_iter = 1, 0

    train_data = CreateDataset(opt)

    if opt.num_gpus > 0:
        torch.cuda.set_device(opt.local_rank)
        # ========================================
        # 1.1. If Windows: change 'nccl' to 'gloo', windows does not support 'nccl'
        # 1.2. If Linux: 'nccl'
        # 2. do not use distributed if only 1 gpu
        # ========================================
        if len(opt.gpu_ids) > 1:
            torch.distributed.init_process_group(
                # 'gloo',
                'nccl',
                init_method='env://'
            )
        device = torch.device(f'cuda:{opt.local_rank}')
    else:
        device = torch.device("cpu")

    # ========================================
    # 1. do not use distributed if only 1 gpu
    # ========================================
    if len(opt.gpu_ids) > 1:
        train_sampler = DistributedSampler(train_data)
        train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                                  num_workers=4, pin_memory=(device.type=="cuda"), sampler=train_sampler)
    else:
        train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                                  num_workers=0, pin_memory=(device.type=="cuda"))


    dataset_size = len(train_loader)
    print('The number of training images = %d' % dataset_size)

    warp_model = AFWM(opt, 45)
    print(warp_model)
    warp_model.to(device)
    warp_model.train()

    # ========================================
    # 1. do not use sync if only 1 gpu
    # ========================================
    if len(opt.gpu_ids) > 1:
        warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

    if opt.isTrain:
        if len(opt.gpu_ids) > 1:
            model = torch.nn.parallel.DistributedDataParallel(warp_model, device_ids=[opt.local_rank])
        else:
            model = warp_model.to(device)  # for CPU or 1 GPU is simple model without DDP


    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()

    params_warp = [p for p in model.parameters()]
    optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))

    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size


    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        # ========================================
        # 1. do not use train_sampler if only 1 gpu
        # ========================================
        if len(opt.gpu_ids) > 1:
            train_sampler.set_epoch(epoch)

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()

            total_steps += 1
            epoch_iter += 1
            save_fake = True

            t_mask = torch.tensor((data['label'].cpu().numpy() == 7), dtype=torch.float32)
            data['label'] = data['label']*(1-t_mask) + t_mask*4
            edge = data['edge']
            pre_clothes_edge = torch.tensor((edge.detach().numpy() > 0.5).astype(np.int32), dtype=torch.float32)
            clothes = data['color']
            clothes = clothes * pre_clothes_edge
            person_clothes_edge = torch.tensor((data['label'].cpu().numpy() == 4), dtype=torch.float32)
            real_image = data['image']
            person_clothes = real_image * person_clothes_edge
            pose = data['pose']
            size = data['label'].size()
            oneHot_size1 = (size[0], 25, size[2], size[3])
            densepose = torch.zeros(oneHot_size1, dtype=torch.float32, device=device)
            densepose = densepose.scatter_(1, data['densepose'].data.long().to(device), 1.0)
            densepose_fore = data['densepose']/24.0
            face_mask = torch.FloatTensor((data['label'].cpu().numpy()==1).astype(np.int32)) + torch.FloatTensor((data['label'].cpu().numpy()==12).astype(np.int32))
            other_clothes_mask = torch.FloatTensor((data['label'].cpu().numpy()==5).astype(np.int32)) + torch.FloatTensor((data['label'].cpu().numpy()==6).astype(np.int32)) + \
                                torch.FloatTensor((data['label'].cpu().numpy()==8).astype(np.int32)) + torch.FloatTensor((data['label'].cpu().numpy()==9).astype(np.int32)) + \
                                torch.FloatTensor((data['label'].cpu().numpy()==10).astype(np.int32))
            preserve_mask = torch.cat([face_mask,other_clothes_mask], 1)
            concat = torch.cat([preserve_mask.to(device), densepose, pose.to(device)], 1)

            flow_out = model(concat.to(device), clothes.to(device), pre_clothes_edge.to(device))
            warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
            warped_prod_edge = x_edge_all[4]

            epsilon = 0.001
            loss_smooth = sum([TVLoss(x) for x in delta_list])
            loss_all = 0

            for num in range(5):
                cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5**(4-num), mode='bilinear')
                cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5**(4-num), mode='bilinear')
                loss_l1 = criterionL1(x_all[num], cur_person_clothes.to(device))
                loss_vgg = criterionVGG(x_all[num], cur_person_clothes.to(device))
                loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.to(device))
                b,c,h,w = delta_x_all[num].shape
                loss_flow_x = (delta_x_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
                loss_flow_x = torch.sum(loss_flow_x) / (b*c*h*w)
                loss_flow_y = (delta_y_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
                loss_flow_y = torch.sum(loss_flow_y) / (b*c*h*w)
                loss_second_smooth = loss_flow_x + loss_flow_y
                loss_all = loss_all + (num+1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num+1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth

            loss_all = 0.01 * loss_smooth + loss_all


            optimizer_warp.zero_grad()
            loss_all.backward()
            optimizer_warp.step()
            ############## Display results and errors ##########

            path = 'sample/' + opt.name
            os.makedirs(path, exist_ok=True)
            if step % 1000 == 0:
                if opt.local_rank == 0:
                    a = real_image.float().to(device)
                    b = person_clothes.to(device)
                    c = clothes.to(device)
                    d = torch.cat([densepose_fore.to(device), densepose_fore.to(device), densepose_fore.to(device)], 1)
                    e = warped_cloth
                    f = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
                    combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0]], 2).squeeze()
                    cv_img=(combine.permute(1,2,0).detach().cpu().numpy() + 1)/2
                    rgb=(cv_img*255).astype(np.uint8)
                    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    cv2.imwrite('sample/'+opt.name+'/'+str(step)+'.jpg', bgr)

            step += 1
            iter_end_time = time.time()
            iter_delta_time = iter_end_time - iter_start_time
            step_delta = (step_per_batch-step%step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
            eta = iter_delta_time*step_delta
            eta = str(datetime.timedelta(seconds=int(eta)))
            time_stamp = datetime.datetime.now()
            now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
            if step % 100 == 0:
                if opt.local_rank == 0:
                    print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch_iter,step, loss_all, eta))

            if epoch_iter >= dataset_size:
                break
       
        # end of epoch 
        iter_end_time = time.time()
        if opt.local_rank == 0:
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        # ========================================
        # 1. model.module => model
        # ========================================
        if epoch % opt.save_epoch_freq == 0:
            if opt.local_rank == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
                save_checkpoint(model, os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch+1)))

        # ========================================
        # 1. model.module => model
        # ========================================
        if epoch > opt.niter:
            model.update_learning_rate(optimizer_warp)
