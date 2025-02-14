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
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
MAX_WAV_VALUE = 32768.0

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        z,log_s1_list, log_s2_list ,log_det_W_list= model_output
        for i, log_s in enumerate(zip(log_s1_list,log_s2_list)):
            if i == 0:
                log_s_total = torch.sum(log_s[0])+torch.sum(log_s[1])
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s[0])+torch.sum(log_s[1])
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total - log_det_W_total
        if not loss <0:
            print("no")
        return loss/(z.size(0)*z.size(1)*z.size(2))


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        #QR分解
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.det(W).abs().log()
            z = self.conv(z)
            return z, log_det_W


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels,
                 kernel_size):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(n_mel_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            # dilation = 1
            # padding = int((kernel_size*dilation - dilation)/2)
            # depthwise = torch.nn.Conv1d(n_channels,n_channels,3,dilation=dilation,padding=padding,groups=n_channels).cuda()
            # pointwise = torch.nn.Conv1d(n_channels,2*n_channels,1).cuda()
            # bn = torch.nn.BatchNorm1d(n_channels)
            # self.in_layers.append(torch.nn.Sequential(bn,depthwise,pointwise))
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)


            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:,spect_offset:spect_offset+2*self.n_channels,:],
                n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:,:self.n_channels,:]
                output = output + res_skip_acts[:,self.n_channels:,:]
            else:
                output = output + res_skip_acts

        return self.end(output)


class WaveGlow(torch.nn.Module):
    def __init__(self, n_mel_channels, n_flows, n_group, n_early_every,
                 n_early_size, WN_config):
        super(WaveGlow, self).__init__()
        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels,
                                                 n_mel_channels,
                                                 1024, stride=256)
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        #self.WN = torch.nn.ModuleList()
        self.WN1 = torch.nn.ModuleList()
        self.WN2 = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(n_group/2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        #12个1*1卷积+仿射组合层
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            #可逆1*1卷积
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            #仿射组合层

            #self.WN.append(WN(n_half, n_mel_channels*n_group, **WN_config))
            self.WN1.append(WN(n_half, n_mel_channels*n_group, **WN_config))
            self.WN2.append(WN(n_half, n_mel_channels * n_group, **WN_config))
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        #6*80*63，6*16000
        spect, audio = forward_input

        #  Upsample spectrogram to size of audio
        # 上采样，扩大音频
        spect = self.upsample(spect)
        #6*80*16896
        #音频和mel谱对齐
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]
        #6*80*16000，对应squeeze操作，16000个采样点8个为一组，保持局部相关性
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)#6*2000*80*8
        #一组mel谱集合起来
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)#6*640*2000
        #squeeze操作，同上
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)#6*8*2000
        output_audio = []
        #log_s_list = []
        log_s1_list = []
        log_s2_list = []
        log_det_W_list = []

        for k in range(self.n_flows):#n_flows=12
            if k % self.n_early_every == 0 and k > 0:#n_early_every=4
                #输出前两个通道
                output_audio.append(audio[:,:self.n_early_size,:])
                audio = audio[:,self.n_early_size:,:]

            audio, log_det_W = self.convinv[k](audio)
            #det|J(f^-1)|=log det|W|
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1)/2)
            #x_a,x_b
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]
            #(logs,t)=WN(x_a,mel),output=[batch_size,8,2000]
            #output = self.WN[k]((audio_0, spect))      
            if spect.type() == 'torch.cuda.HalfTensor':
                # input_0 = torch.from_numpy(np.ones(audio_0.size())/MAX_WAV_VALUE).float().cuda().half()
                input_0 = torch.from_numpy(np.zeros(audio_0.size())/MAX_WAV_VALUE).float().cuda().half()
                input_0 = torch.cuda.HalfTensor(input_0)
                input_0 = torch.autograd.Variable(input_0)
                # input_0 = torch.cuda.HalfTensor(audio_0.size()).normal_()/MAX_WAV_VALUE
            else:
                # input_0 = torch.from_numpy(np.ones(audio_0.size()) / MAX_WAV_VALUE).float().cuda()
                input_0 = torch.from_numpy(np.zeros(audio_0.size()) / MAX_WAV_VALUE).float().cuda()
                input_0 = torch.autograd.Variable(input_0)
                # input_0 = torch.cuda.FloatTensor(audio_0.size()).normal_()/MAX_WAV_VALUE
            output1 = self.WN1[k]((input_0, spect))
            log_s1 = output1[:, n_half:, :]
            t_1 = output1[:, :n_half, :]
            y_1 =  torch.exp(log_s1)*audio_0+t_1
            output2 = self.WN2[k](((y_1+audio_0), spect))
            log_s2 = output2[:, n_half:, :]
            t_2 = output2[:, :n_half, :]
            y_2 = torch.exp(log_s2)*audio_1+t_2
            #y_1 = 
            #log_s2 = self.WN2[k]((y_1, spect))
            #y_2 = torch.exp(log_s2)*audio_1+self.WN4[k](y_1,spect)
            #前一半仿射s，后一半仿射t
            #log_s = output[:, n_half:, :]
            #b = output[:, :n_half, :]
            #x_b'=s*x_b+t
            #audio_1 = torch.exp(log_s)*audio_1 + b
            #记录logs
            #log_s_list.append(log_s)
            log_s1_list.append(log_s1)
            log_s2_list.append(log_s2)
            #concat(x_a,x_b')
            #audio = torch.cat([audio_0, audio_1],1)
            audio = torch.cat([y_1,y_2],1)

        output_audio.append(audio)
        return torch.cat(output_audio,1),  log_s1_list, log_s2_list, log_det_W_list

    def infer(self, spect, sigma=1.0):
        #一维反卷积
        #1*80*375
        spect = self.upsample(spect)
        #1*80*96768
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]
        #1*80*96000
        #unfold滑动窗口，permute换维
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        #1*12000*80*8
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        #1*640*12000，True
        # y_0 = torch.from_numpy(np.ones([spect.size(0),
        #                                   int(self.n_remaining_channels/2),
        #                                   spect.size(2)])).cuda()/MAX_WAV_VALUE
        y_0 = torch.from_numpy(np.zeros([spect.size(0),
                                          int(self.n_remaining_channels/2),
                                          spect.size(2)])).cuda()/MAX_WAV_VALUE
        if spect.type() == 'torch.cuda.HalfTensor':
            audio = torch.cuda.HalfTensor(spect.size(0),
                                          self.n_remaining_channels,
                                          spect.size(2)).normal_()
            y_0 =y_0.half()
            y_0 = torch.cuda.HalfTensor(y_0)
            y_0 = torch.autograd.Variable(sigma * y_0)
            # y_0 = torch.cuda.HalfTensor(spect.size(0),
            #                               int(self.n_remaining_channels/2),
            #                               spect.size(2)).normal_()/MAX_WAV_VALUE
            # y_0 = sigma * y_0
            #1*4*12000
        else:
            audio = torch.cuda.FloatTensor(spect.size(0),
                                           self.n_remaining_channels,
                                           spect.size(2)).normal_()
            y_0 =y_0.float()
            y_0 = torch.autograd.Variable(sigma * y_0)
            # y_0 = torch.cuda.FloatTensor(spect.size(0),
            #                               int(self.n_remaining_channels/2),
            #                               spect.size(2)).normal_()/MAX_WAV_VALUE
            # y_0 = sigma * y_0
        #封装数据
        audio = torch.autograd.Variable(sigma*audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            # 1*2*12000
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]
            #1*4*12000
            #output = self.WN[k]((audio_0, spect))
            output1 = self.WN1[k]((y_0, spect))
            log_s1 = output1[:, n_half:, :]
            t_1 = output1[:, :n_half, :]
            y_1 = audio_0
            x_a = (y_1-t_1)/torch.exp(log_s1)
            
            output2 = self.WN2[k](((y_1+audio_0)/2, spect))
            log_s2 = output2[:, n_half:, :]
            t_2 = output2[:, :n_half, :]
            y_2 = audio_1
            x_b = (y_2-t_2)/torch.exp(log_s2)
            #1*2*12000
            #s = output[:, n_half:, :]
            #b = output[:, :n_half, :]
            #1*2*12000,(y_b-t)/s
            #audio_1 = (audio_1 - b)/torch.exp(s)
            #1*4*12000
            audio = torch.cat([x_a, x_b],1)
            #1*1卷积，4*4
            audio = self.convinv[k](audio, reverse=True)
            #1*4*12000,每经过四个flows就加入两个channel
            if k % self.n_early_every == 0 and k > 0:
                # y_0 = torch.from_numpy(np.ones([spect.size(0),
                #                                 int((audio.size()[1]+self.n_early_size) / 2),
                #                                 spect.size(2)])).cuda() / MAX_WAV_VALUE
                y_0 = torch.from_numpy(np.zeros([spect.size(0),
                                          int((audio.size()[1]+self.n_early_size) / 2),
                                          spect.size(2)])).cuda()/MAX_WAV_VALUE
                if spect.type() == 'torch.cuda.HalfTensor':
                    z = torch.cuda.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                    y_0 = y_0.half()
                    y_0 = torch.cuda.HalfTensor(y_0)
                    y_0 = torch.autograd.Variable(sigma*y_0)
                    # y_0 = torch.cuda.HalfTensor(spect.size(0),
                    #                             int((audio.size()[1]+self.n_early_size) / 2),
                    #                             spect.size(2)).normal_()/MAX_WAV_VALUE
                    # y_0 = sigma * y_0
                else:
                    z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                    y_0 = y_0.float()
                    y_0 = torch.autograd.Variable(sigma*y_0)
                    # y_0 = torch.cuda.FloatTensor(spect.size(0),
                    #                             int((audio.size()[1]+self.n_early_size) / 2),
                    #                             spect.size(2)).normal_()/MAX_WAV_VALUE
                    # y_0 = sigma * y_0
                audio = torch.cat((sigma*z, audio),1)
                #k=8,1*6*12000，k=4,1*8*12000
        #1*8*12000
        audio = audio.permute(0,2,1).contiguous().view(audio.size(0), -1).data
        #1*96000
        
        return audio

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        # for WN in waveglow.WN:
        #     WN.start = torch.nn.utils.remove_weight_norm(WN.start)#？移除权重归一化
        #     WN.in_layers = remove(WN.in_layers)
        #     WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
        #     WN.res_skip_layers = remove(WN.res_skip_layers)
        for WN in waveglow.WN1:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)#？移除权重归一化
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove(WN.res_skip_layers)

        for WN in waveglow.WN2:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)#？移除权重归一化
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow

def fuse_conv_and_bn(conv, bn):
    fusedconv = torch.nn.Conv1d(
            conv.in_channels,
            conv.out_channels,
            kernel_size = conv.kernel_size,
            padding=conv.padding,
            bias=True,
            groups=conv.groups)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    w_bn = w_bn.clone()
    fusedconv.weight.data = torch.mm(w_bn, w_conv).view(fusedconv.weight.size())
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros( conv.weight.size(0) )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    b_bn = torch.unsqueeze(b_bn, 1)
    bn_3 = b_bn.expand(-1, 3)
    b = torch.matmul(w_conv, torch.transpose(bn_3, 0, 1))[range(b_bn.size()[0]), range(b_bn.size()[0])]
    fusedconv.bias.data = ( b_conv + b )
    return fusedconv

def remove_batch_norm(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        depthwise = fuse_conv_and_bn(old_conv[1], old_conv[0])
        pointwise = old_conv[2]
        new_conv_list.append(torch.nn.Sequential(depthwise, pointwise))
    return new_conv_list

def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)#？移除权重归一化
        new_conv_list.append(old_conv)
    return new_conv_list
