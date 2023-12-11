# Revisiting-De-Identification

Our code is modified from eznlp.
* Check [eznlp](https://github.com/syuoni/eznlp) for more details


## Installation
### Create an environment
```bash
$ conda create --name eznlp python=3.8
$ conda activate eznlp
```

### Install dependencies
```bash
$ conda install numpy=1.18.5 pandas=1.0.5 xlrd=1.2.0 matplotlib=3.2.2 
$ conda install pytorch=1.7.1 torchvision=0.8.2 torchtext=0.8.1 {cpuonly|cudatoolkit=10.2|cudatoolkit=11.0} -c pytorch 
$ pip install -r requirements.txt 
```

### Install `eznlp`
* From source (recommended)
```bash
$ python setup.py sdist
$ pip install dist/eznlp-<version>.tar.gz --no-deps
```

* With `pip`
```bash
$ pip install eznlp --no-deps
```

* Replace  

Then we substitute the [./eznlp/] file for [./anaconda3/envs/eznlp/lib/python3.8/site-packages/eznlp/] file


## Data DownLoad
### HwaMei-Privacy

HwaMei-Privacy is the first de-identification task for Chinese electronic medical records (EMRs). It provides three datasets, i.e., HM, SY, and CCKS. Check our paper for more details. 

The released data have been manually *de-identified*. Specifically, we have carefully replaced the protected health information (PHI) mentions by realistic *surrogates* (Stubbs et al., 2015). For example, all the person names are replaced by combinations of randomly sampled family and given names, where the sampling accords to the frequencies reported by National Bureau of Statistics of China. All the locations are replaced by randomly sampled addresses in China. (In other words, all the PHI mentions are *fake* in the released data.) Such process preserves the usability of our data and prevent PHI leak simultaneously. 

HwaMei-Privacy is available upon request. Please visit this [link](http://47.99.121.158:8000), sign and upload the data use agreement. Please strictly abide by the terms of the agreement. Contact liuyiyang@ucas.ac.cn if you need help. 

*Then put the datasets at [./scripts/data/].
## Running the Code

```bash
$ python scripts/identification_recognition.py --dataset HwaMei_Privacy [options]
```



## Citation
If you find our code useful, please cite the following papers: 
```
@inproceedings{liu-etal-2023-revisiting-de,
    title = "Revisiting De-Identification of Electronic Medical Records: Evaluation of Within- and Cross-Hospital Generalization",
    author = "Liu, Yiyang  and
      Li, Jinpeng  and
      Zhu, Enwei",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.224",
    pages = "3666--3674",
    abstract = "The de-identification task aims to detect and remove the protected health information from electronic medical records (EMRs). Previous studies generally focus on the within-hospital setting and achieve great successes, while the cross-hospital setting has been overlooked. This study introduces a new de-identification dataset comprising EMRs from three hospitals in China, creating a benchmark for evaluating both within- and cross-hospital generalization. We find significant domain discrepancy between hospitals. A model with almost perfect within-hospital performance struggles when transferred across hospitals. Further experiments show that pretrained language models and some domain generalization methods can alleviate this problem. We believe that our data and findings will encourage investigations on the generalization of medical NLP models.",
}
```

```
@inproceedings{zhu2023deep,
  title={Deep Span Representations for Named Entity Recognition},
  author={Zhu, Enwei and Liu, Yiyang and Li, Jinpeng},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  month={jul},
  year={2023},
  address={Toronto, Canada},
  publisher={Association for Computational Linguistics},
  url={https://aclanthology.org/2023.findings-acl.672},
  doi={10.18653/v1/2023.findings-acl.672},
  pages={10565--10582}
}
```

```
@inproceedings{zhu2022boundary,
  title={Boundary Smoothing for Named Entity Recognition},
  author={Zhu, Enwei and Li, Jinpeng},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month={may},
  year={2022},
  address={Dublin, Ireland},
  publisher={Association for Computational Linguistics},
  url={https://aclanthology.org/2022.acl-long.490},
  doi={10.18653/v1/2022.acl-long.490},
  pages={7096--7108}
}
```

```
@article{zhu2023framework,
  title={A unified framework of medical information annotation and extraction for {C}hinese clinical text},
  author={Zhu, Enwei and Sheng, Qilin and Yang, Huanwan and Liu, Yiyang and Cai, Ting and Li, Jinpeng},
  journal={Artificial Intelligence in Medicine},
  volume={142},
  pages={102573},
  year={2023},
  publisher={Elsevier}
}
```

## References
* Zhu, E., Li, J. Boundary Smoothing for Named Entity Recognition. In *ACL 2022*. 
* Zhu, E., Sheng, Q., Yang, H., Liu, Y., Cai, T., and Li, J. A unified framework of medical information annotation and extraction for Chinese clinical text. *Artificial Intelligence in Medicine*, 2023, 142:102573.
* Zhu, E., Liu, Y., and Li, J. Deep Span Representations for Named Entity Recognition. In *ACL 2023*.
