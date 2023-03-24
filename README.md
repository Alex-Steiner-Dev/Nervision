# Nervision: Text to 3D Model AI using PyTorch

## Introduction

Nervision is a Text to 3D Model AI developed by Alex Steiner for Acceleration Lab at the International School of Treviso H-FARM. The aim of this project is to generate 3D models from textual descriptions using PyTorch. 

To develop this project, I utilized a research paper created by the University of Hong Kong and Stanford University, as well as MIT. Additionally, to generate the texture, I used a pre-made project called "TANGO" developed by a Gorilla-Lab-SCUT.

## Requirements

To run Nervision, the following requirements must be met:

- PyTorch
- Keras
- Python 3.x
- CUDA-enabled GPU (optional)
- Open 3D
- Trimesh

## Usage

1. Clone the Nervision repository to your local machine.
2. Install the required packages using `pip install -r requirements.txt`
3. Run the program using `./run.sh`
4. Input a textual description to generate a 3D model.

## Acknowledgments

- University of Hong Kong
- Stanford University
- MIT
- TANGO - Gorilla-Lab-SCUT

## Conclusion

In conclusion, Nervision is a Text to 3D Model AI developed by Alex Steiner for Acceleration Lab at the International School of Treviso H-FARM. The project aims to generate 3D models from textual descriptions using PyTorch. The project utilized a research paper created by the University of Hong Kong and Stanford University, as well as MIT. To generate the texture, a pre-made project called "TANGO" developed by Gorilla-Lab-SCUT was used. 

Nervision is a powerful tool for generating 3D models quickly and easily from textual descriptions. It has potential applications in various industries, including architecture, gaming, and film.

## Citation
```
@misc{tang2022warpinggan,
      title={WarpingGAN: Warping Multiple Uniform Priors for Adversarial 3D Point Cloud Generation}, 
      author={Yingzhi Tang and Yue Qian and Qijian Zhang and Yiming Zeng and Junhui Hou and Xuefei Zhe},
      year={2022},
      eprint={2203.12917},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@inproceedings{ChenChenNeurIPS22,
  title={TANGO: Text-driven Photorealistic and Robust 3D Stylization via Lighting Decomposition},
  author={Yongwei Chen and Rui Chen and Jiabao Lei and Yabin Zhang and Kui Jia},
  booktitle={Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
