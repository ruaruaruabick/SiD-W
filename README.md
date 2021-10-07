
## SiD-Waveflow


This work is the implemention of SiD-Waveflow.

Visit our [website] for audio samples.

## Setup

1. Clone our repo and initialize submodule

   ```command
   git clone https://github.com/NVIDIA/waveglow.git
   cd waveglow
   git submodule init
   git submodule update
   ```

2. Install requirements `pip3 install -r requirements.txt`

3. Install [Apex]


## Train your own model

1. Download [CSMSC]. In this example it's in `~/BBdata/`


2. Train 

   ```command
   mkdir checkpoints
   python train.py -c config.json
   ```

   For mixed precision training set `"fp16_run": true` on `config.json`.

3. Make test set mel-spectrograms

   `python mel2samp.py -f traintestset_chn/test_files_copy.txt -o ./inferaudio/chn_mel -c config.json`

5. Do inference with your network

   ```command
   ls inferaudio/chn_mel/*.pt > mel_files.txt
   python3 inference.py -f mel_files.txt -w checkpoints/test1_chn_model -o ./inferaudio --is_fp16 -s 0.6
   ```

[//]: # (TODO)
[//]: # (PROVIDE INSTRUCTIONS FOR DOWNLOADING LJS)
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://ruaruaruabick.github.io/
[paper]: https://arxiv.org/abs/1811.00002
[WaveNet implementation]: https://github.com/r9y9/wavenet_vocoder
[Glow]: https://blog.openai.com/glow/
[WaveNet]: https://deepmind.com/blog/wavenet-generative-model-raw-audio/
[PyTorch]: http://pytorch.org
[published model]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[mel-spectrograms]: https://drive.google.com/file/d/1g_VXK2lpP9J25dQFhQwx7doWl_p20fXA/view?usp=sharing
[LJ Speech Data]: https://keithito.com/LJ-Speech-Dataset
[Apex]: https://github.com/nvidia/apex
[CSMSC]:https://www.data-baker.com/open_source.html