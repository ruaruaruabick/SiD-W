import python_speech_features as psf
import numpy as np
import torch

import spafe.utils.vis as vis
import spafe.features.rplp as sfp
class get_mfcc: 
    def __init__(self) -> None:
        self.numc = 13
    def get_mfcc(self, sample_rate, signal):
        audio_norm = signal.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        
        wav_feature = psf.mfcc(audio_norm, sample_rate, numcep=self.numc, winlen=0.025, winstep=0.01,
                           nfilt=26, nfft=1024, lowfreq=0, highfreq=None, preemph=0.97)
        
        '''
        signal - 需要用来计算特征的音频信号，应该是一个N*1的数组
        samplerate - 我们用来工作的信号的采样率
        winlen - 分析窗口的长度，按秒计，默认0.025s(25ms)
        winstep - 连续窗口之间的步长，按秒计，默认0.01s（10ms）
        numcep - 倒频谱返回的数量，默认13
        nfilt - 滤波器组的滤波器数量，默认26
        nfft - FFT的大小，默认512
        lowfreq - 梅尔滤波器的最低边缘，单位赫兹，默认为0
        highfreq - 梅尔滤波器的最高边缘，单位赫兹，默认为采样率/2
        preemph - 应用预加重过滤器和预加重过滤器的系数，0表示没有过滤器，默认0.97
        ceplifter - 将升降器应用于最终的倒谱系数。 0没有升降机。默认值为22。
        appendEnergy - 如果是true，则将第0个倒谱系数替换为总帧能量的对数。 
        '''
        d_mfcc_feat = psf.delta(wav_feature, 1)
        d_mfcc_feat2 = psf.delta(wav_feature, 2)
        feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
        mfcck = feature
        m, n = mfcck.shape
        mfcck = torch.from_numpy(mfcck).permute(1,0)
        torch.squeeze(mfcck, 0)
        return mfcck
class get_plp:
    def __init__(self) -> None:
        self.numc = 13
    def get_plp(self, sample_rate, signal):
        #audio_norm = signal.unsqueeze(0)
        #audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        audio_norm = signal.numpy()
        wav_feature = sfp.plp(sig=audio_norm,fs=sample_rate)
        '''
        signal - 需要用来计算特征的音频信号，应该是一个N*1的数组
        samplerate - 我们用来工作的信号的采样率
        winlen - 分析窗口的长度，按秒计，默认0.025s(25ms)
        winstep - 连续窗口之间的步长，按秒计，默认0.01s（10ms）
        numcep - 倒频谱返回的数量，默认13
        nfilt - 滤波器组的滤波器数量，默认26
        nfft - FFT的大小，默认512
        lowfreq - 梅尔滤波器的最低边缘，单位赫兹，默认为0
        highfreq - 梅尔滤波器的最高边缘，单位赫兹，默认为采样率/2
        preemph - 应用预加重过滤器和预加重过滤器的系数，0表示没有过滤器，默认0.97
        ceplifter - 将升降器应用于最终的倒谱系数。 0没有升降机。默认值为22。
        appendEnergy - 如果是true，则将第0个倒谱系数替换为总帧能量的对数。 
        '''
        d_mfcc_feat = psf.delta(wav_feature, 1)
        d_mfcc_feat2 = psf.delta(wav_feature, 2)
        feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
        mfcck = feature
        m, n = mfcck.shape
        mfcck = torch.from_numpy(mfcck).permute(1,0)
        torch.squeeze(mfcck, 0)
        return mfcck 
class get_logfbank:
    def __init__(self) -> None:
        self.numc = 13
    def get_logfbank(self, sample_rate, signal):
        audio_norm = signal.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        
        wav_feature = psf.logfbank(audio_norm, sample_rate,nfilt=40,nfft=1024)
        '''
        signal - 需要用来计算特征的音频信号，应该是一个N*1的数组
        samplerate - 我们用来工作的信号的采样率
        winlen - 分析窗口的长度，按秒计，默认0.025s(25ms)
        winstep - 连续窗口之间的步长，按秒计，默认0.01s（10ms）
        numcep - 倒频谱返回的数量，默认13
        nfilt - 滤波器组的滤波器数量，默认26
        nfft - FFT的大小，默认512
        lowfreq - 梅尔滤波器的最低边缘，单位赫兹，默认为0
        highfreq - 梅尔滤波器的最高边缘，单位赫兹，默认为采样率/2
        preemph - 应用预加重过滤器和预加重过滤器的系数，0表示没有过滤器，默认0.97
        ceplifter - 将升降器应用于最终的倒谱系数。 0没有升降机。默认值为22。
        appendEnergy - 如果是true，则将第0个倒谱系数替换为总帧能量的对数。 
        '''
        #d_mfcc_feat = psf.delta(wav_feature, 1)
        #d_mfcc_feat2 = psf.delta(wav_feature, 2)
        #feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
        mfcck = wav_feature
        m, n = mfcck.shape
        mfcck = torch.from_numpy(mfcck).permute(1,0)
        torch.squeeze(mfcck, 0)
        return mfcck 