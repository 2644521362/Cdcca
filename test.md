<h1>Cloud-Device Collaborative Learning for Multimodal Large Language Models</h1>

## Intro
Official implementation of ['Cloud-Device Collaborative Learning for Multimodal Large Language Models'](https://arxiv.org/pdf/2312.16279).

This repository presents **CDCCA**, a Self-Corrected Multimodal Large Language Model designed to optimize the performance of models deployed on client devices by leveraging advanced cloud capabilities. 
 🔥
## Abstract
The burgeoning field of Multimodal Large Language Models (MLLMs) has exhibited remarkable performance in diverse tasks such as captioning, commonsense reasoning, and visual scene understanding. However, the deployment of these large-scale MLLMs on client devices is hindered by their extensive model parameters, leading to a notable decline in generalization capabilities when these models are compressed for device deployment. Addressing this challenge, we introduce a Cloud-Device Collaborative Continual Adaptation framework, designed to enhance the performance of compressed, device-deployed MLLMs by leveraging the robust capabilities of cloud-based, larger-scale MLLMs.
Our framework is structured into three key components: a device-to-cloud uplink for efficient data transmission, cloud-based knowledge adaptation, and an optimized cloud-to-device downlink for model deployment. In the uplink phase, we employ an Uncertainty-guided Token Sampling (UTS) strategy to effectively filter out-of-distribution tokens, thereby reducing transmission costs and improving training efficiency. On the cloud side, we propose Adapter-based Knowledge Distillation (AKD) method to transfer refined knowledge from large-scale to compressed, pocket-size MLLMs. Furthermore, we propose a Dynamic Weight update Compression (DWC) strategy for the downlink, which adaptively selects and quantizes updated weight parameters, enhancing transmission efficiency and reducing the representational disparity between cloud and device models. Extensive experiments on several multimodal benchmarks demonstrate the superiority of our proposed framework over prior Knowledge Distillation and device-cloud collaboration methods. Notably, we also validate the feasibility of our approach to real-world experiments.


## Contributors
**Authors:**
- Guanqun Wang<sup>1*</sup>, Jiaming Liu<sup>1*</sup>, Chenxuan Li<sup>1*</sup>, Yuan Zhang<sup>1</sup>, Junpeng Ma<sup>1</sup>, Xinyu Wei<sup>1</sup>, Kevin Zhang<sup>1</sup>
- Maurice Chong<sup>1</sup>, Renrui Zhang<sup>2</sup>, Yijiang Liu<sup>3</sup>, Shanghang Zhang<sup>1†</sup>

**Affiliations:**
- <sup>1</sup>National Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University
- <sup>2</sup>Shanghai AI Lab
- <sup>3</sup>Nanjing University



## Citation
If you find our CDCCA code and paper useful, hope you can cite our article:
```bash
@article{wang2023cloud,
  title={Cloud-Device Collaborative Learning for Multimodal Large Language Models},
  author={Wang, Guanqun and Liu, Jiaming and Li, Chenxuan and Ma, Junpeng and Zhang, Yuan and Wei, Xinyu and Zhang, Kevin and Chong, Maurice and Zhang, Ray and Liu, Yijiang and others},
  journal={arXiv preprint arXiv:2312.16279},
  year={2023}
}
```
