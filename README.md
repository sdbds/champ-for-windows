
<h1 align='Center'>Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance</h1>

<div align='Center'>
    <a href='https://github.com/ShenhaoZhu' target='_blank'>Shenhao Zhu</a><sup>*1</sup>&emsp;
    <a href='https://github.com/Leoooo333' target='_blank'>Junming Leo Chen</a><sup>*2</sup>&emsp;
    <a href='https://github.com/daizuozhuo' target='_blank'>Zuozhuo Dai</a><sup>3</sup>&emsp;
    <a href='https://ai3.fudan.edu.cn/info/1088/1266.htm' target='_blank'>Yinghui Xu</a><sup>2</sup>&emsp;
    <a href='https://yoyo000.github.io/' target='_blank'>Yao Yao</a><sup>1</sup>&emsp;
    <a href='http://zhuhao.cc/home/' target='_blank'>Hao Zhu</a><sup>+1</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>+2</sup>
</div>
<div align='Center'>
    <sup>1</sup>Nanjing University <sup>2</sup>Fudan University <sup>3</sup>Alibaba Group
</div>
<div align='Center'>
    <sup>*</sup>Equal Contribution
    <sup>+</sup>Corresponding Author
</div>

<div align='Center'>
    <a href='https://fudan-generative-vision.github.io/champ/#/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2403.14781'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://youtu.be/2XVsy9tQRAY'><img src='https://badges.aleen42.com/src/youtube.svg'></a>
</div>

https://github.com/fudan-generative-vision/champ/assets/82803297/b4571be6-dfb0-4926-8440-3db229ebd4aa

# Framework
![framework](assets/framework.jpg)

# ⚒️ Installation

prerequisites: `3.11>=python>=3.8`, `CUDA>=11.3`, `ffmpeg` and `git`.

Python and Git:

- Python 3.10.11: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
- git: https://git-scm.com/download/win

- Install [ffmpeg](https://ffmpeg.org/) for your operating system
  (https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)
  
  notice:step 4 use windows system Set Enviroment Path.

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

```
git clone --recurse-submodules https://github.com/sdbds/champ-for-windows/
```

Install with Powershell run `install.ps1` or `install-cn.ps1`(for Chinese)

### Use local model

Add loading local safetensors or ckpt,you can change `configs/inference.yaml` about `base_model_path` for your local SD1.5 model.
such as `"D:\\stablediffusion-webui\\models\\Stable-diffusion\\v1-5-pruned.ckpt"`

## No need Download models manually

~~# Download pretrained models~~

~~1. Download pretrained weight of base models:~~ 
    ~~- [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)~~
    ~~- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)~~
    ~~- [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)~~

~~2. Download our checkpoints:~~
~~Our [checkpoints](https://drive.google.com/drive/folders/1hZiOHG-qDf0Pj7tvfxC70JQ6wHUvUDoY?usp=sharing) consist of denoising UNet, guidance encoders, Reference UNet, and motion module.~~

~~Finally, these pretrained models should be organized as follows:~~

```text
./pretrained_models/
|-- champ
|   |-- denoising_unet.pth
|   |-- guidance_encoder_depth.pth
|   |-- guidance_encoder_dwpose.pth
|   |-- guidance_encoder_normal.pth
|   |-- guidance_encoder_semantic_map.pth
|   |-- reference_unet.pth
|   `-- motion_module.pth
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `-- v1-inference.yaml
```

# Inference
We have provided several sets of [example data](https://drive.google.com/file/d/1-sJlnnZu-nTNTvRtvFVr_y-2_CA5-_Yz/view?usp=sharing) for inference. Please first download and place them in the `example_data` folder. 

Here is the command for inference:
```bash
  python inference.py --config configs/inference.yaml
```

or Powershell run with `run_inference.ps1`

Animation results will be saved in `results` folder. You can change the reference image or the guidance motion by modifying `inference.yaml`. 

You can also extract the driving motion from any videos and then render with Blender. We will later provide the instructions and scripts for this.

Note: The default motion-01 in `inference.yaml` has more than 500 frames and takes about 36GB VRAM. If you encounter VRAM issues, consider switching to other example data with less frames.

# Acknowledgements
We thank the authors of [MagicAnimate](https://github.com/magic-research/magic-animate), [Animate Anyone](https://github.com/HumanAIGC/AnimateAnyone), and [AnimateDiff](https://github.com/guoyww/AnimateDiff) for their excellent work. Our project is built upon [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), and we are grateful for their open-source contributions.

# Citation
If you find our work useful for your research, please consider citing the paper:
```
@misc{zhu2024champ,
      title={Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance}, 
      author={Shenhao Zhu and Junming Leo Chen and Zuozhuo Dai and Yinghui Xu and Xun Cao and Yao Yao and Hao Zhu and Siyu Zhu},
      year={2024},
      eprint={2403.14781},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Opportunities available
Multiple research positions are open at the **Generative Vision Lab, Fudan University**! Include:
* Research assistant
* Postdoctoral researcher
* PhD candidate
* Master students

Interested individuals are encouraged to contact us at [siyuzhu@fudan.edu.cn](mailto://siyuzhu@fudan.edu.cn) for further information.
