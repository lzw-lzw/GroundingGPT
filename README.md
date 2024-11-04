# GroundingGPT: Language-Enhanced Multi-modal Grounding Model

<a href='https://lzw-lzw.github.io/GroundingGPT.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2401.06071'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![](https://img.shields.io/badge/Datasets-GroundingGPT-yellow)](https://huggingface.co/datasets/zwli/GroundingGPT) [![](https://img.shields.io/badge/Models-GroundingGPT-yellow)](https://huggingface.co/zwli/GroundingGPT)   


## Introduction
GroundingGPT is an end-to-end multimodal grounding model that accurately comprehends inputs and possesses robust grounding capabilities across multi modalities,including images, audios, and videos. To address the issue of limited data, we construct a diverse and high-quality multimodal training dataset. This dataset encompasses a rich collection of multimodal data enriched with spatial and temporal information, thereby serving as a valuable resource to foster further advancements in this field. Extensive experimental evaluations validate the effectiveness of the GroundingGPT model in understanding and grounding tasks across various modalities. 

More details are available in our [project page](https://lzw-lzw.github.io/GroundingGPT.github.io/). 

<p align="center">
    <img src="images/architecture.png" width="80%"> <br>
    The overall structure of GroundingGPT. Blue boxes represent video as input, while yellow boxes represent image as input.
</p>

## News
* **[2024.5]**  Our paper is accepted to ACL 2024!
* **[2024.4]**  Our [model](https://huggingface.co/zwli/GroundingGPT) is available now!
* **[2024.3]**  Our [training dataset](https://huggingface.co/datasets/zwli/GroundingGPT) are available now!
* **[2024.3]**  Our code are available now! 

## Dependencies and Installation
        git clone https://github.com/lzw-lzw/GroundingGPT.git
        cd GroundingGPT
        conda create -n groundinggpt python=3.10 -y
        conda activate groundinggpt
        pip install -r requirements.txt 
        pip install flash-attn --no-build-isolation


## Training
### Training model preparation
- Put the prepared checkpoints in directory `./ckpt`.
- Prepare ImageBind checkpoint: download [imagebind_huge.pth](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) in link and put it under directory `./ckpt/imagebind`.
- Prepare blip2 checkpoint: download [blip2_pretrained_flant5xxl.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth) in link and put it under directory `./ckpt`.
  
### Training dataset preparation
- Please put the prepared checkpoints in file `dataset`.
- Prepare LLaVA, COCO, GQA, OCR-VQA, TextVQA, VisualGenome datasets: follow [LLaVA](https://github.com/haotian-liu/LLaVA).
- Prepare Flickr30K-Entities datasets: follow [Flickr30K-Entities](https://bryanplummer.com/Flickr30kEntities/).
- Prepare Valley datasets: follow [Valley](https://github.com/RupertLuo/Valley).
- Prepare DiDeMO datasets: follow [DiDeMO](https://github.com/LisaAnne/TemporalLanguageRelease).
- Prepare ActivityNet Captions datasets: follow [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/).
- Prepare Charades-STA datasets: follow [Charades-STA](https://github.com/jiyanggao/TALL).
- Prepare VGGSS datasets: follow [VGGSS](https://www.robots.ox.ac.uk/~vgg/research/lvs/).
- Prepare WaveCaps datasets: follow [WaveCaps](https://github.com/XinhaoMei/WavCaps).
- Prepare Clotho datasets: follow [Clotho](https://zenodo.org/records/3490684).


### Training

-

## Inference

- Download [GroundingGPT-7B](https://huggingface.co/zwli/GroundingGPT) and change the model_path in `GroundingGPT/lego/serve/cli.py`
- Use the script to inference

        python3 lego/serve/cli.py


## Demo
- Download [GroundingGPT-7B](https://huggingface.co/zwli/GroundingGPT) and change the model_path in line 141 of `GroundingGPT/lego/serve/gradio_web_server.py`
- Use the script to launch a gradio web demo

        python3 lego/serve/gradio_web_server.py


## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)
- [Shikra](https://github.com/shikras/shikra)

### Citation
If you find GroundingGPT useful for your your research and applications, please cite using this BibTeX:
    
    @inproceedings{li2024groundinggpt,
      title={Groundinggpt: Language enhanced multi-modal grounding model},
      author={Li, Zhaowei and Xu, Qi and Zhang, Dong and Song, Hang and Cai, Yiqing and Qi, Qi and Zhou, Ran and Pan, Junting and Li, Zefeng and Tu, Vu and others},
      booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages={6657--6678},
      year={2024}
    }
