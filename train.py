# -*- coding: utf-8 -*
# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import copy
import argparse
import json
import os
import torch
import numpy as np
#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from glow import WaveGlow, WaveGlowLoss
from mel2samp import Mel2Samp

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, schedular,learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveGlow(**waveglow_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                 'iteration': iteration,
                 'optimizer': optimizer.state_dict(),
                 'learning_rate': learning_rate,
                 'schedular':schedular
                }, filepath)
def validate(model,criterion,valset,epoch,batch_size,n_gpus,rank,output_directory,logger):
    model.eval()
    with torch.no_grad():
        test_sampler = DistributedSampler(valset) if n_gpus > 1 else None
        test_loader = DataLoader(valset, num_workers=1, shuffle=False,
                              sampler=test_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)
        val_loss =[]
        for i,batch in enumerate(test_loader):
            model.zero_grad()
            #mel=batch*80*63,batch*16000
            mel, audio = batch
            #封装数据
            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.cuda())
            outputs = model((mel, audio))
            #计算loss
            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            val_loss.append(reduced_loss)
        logger.add_scalar('test_loss', np.mean(val_loss), epoch)

def train(num_gpus, rank, group_name,tnum, output_directory, epochs, learning_rate,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard):
    #设定随机数以便复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======
    #计算Loss
    criterion = WaveGlowLoss(sigma)
    #构建waveglow模型
    model = WaveGlow(**waveglow_config).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param", pytorch_total_params)
    print("param trainable", pytorch_total_params_train)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======
    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #apex加速
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        #iteration = checkpoint_dict['iteration']
        #optimizer.load_state_dict(checkpoint_dict['optimizer'])
        model_for_loading = checkpoint_dict['model']
        model.load_state_dict(model_for_loading.state_dict())
        print("Loaded checkpoint '{}' (iteration {})".format(
            checkpoint_path, iteration))
        #model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
        #                                               optimizer)
        iteration += 1  # next iteration is iteration + 1
    temp_config = copy.deepcopy(data_config)
    temp_config['training_files'] = data_config['training_files'].replace('1',str(tnum))
    trainset = Mel2Samp(**data_config)
    testconfig = copy.deepcopy(data_config)
    testconfig["training_files"] = "traintestset_eng/test_files_eng.txt"
    testset = Mel2Samp(**testconfig)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)
    #用不到
    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, 'logs'))

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 5e-5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.25)
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            #梯度置0，z符合高斯0分布
            model.zero_grad()
            #mel=batch*80*63,batch*16000
            mel, audio = batch
            #封装数据
            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.cuda())
            outputs = model((mel, audio))
            #计算loss
            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            #apex加速还原
            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            
            if not reduced_loss < 0:
                print("no")
            print("{}:\t{:.9f}".format(iteration, reduced_loss))
            if with_tensorboard and rank == 0:
                logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)

            if (iteration % iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_path = "{}/waveglow_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, scheduler,learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1
            # num_p = 0
            # for param in model.parameters():
            #     num_p += param.numel()
            # print(num_p)
        #scheduler.step()
        # validate
        if rank == 0:
            validate(model,criterion,testset,epoch,batch_size,num_gpus,rank,output_directory,logger)
            model.train()
    checkpoint_path = "{}/test{}_eng_model".format(
                            output_directory, tnum)
    save_checkpoint(model, optimizer, scheduler, learning_rate, iteration,
                    checkpoint_path)
if __name__ == "__main__":
    #解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")
    #自动使用高效算法
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    for i in range(1,2):
        tnum=i
        train(num_gpus, args.rank, args.group_name,tnum, **train_config)
