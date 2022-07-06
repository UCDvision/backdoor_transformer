# Backdoor Attacks on Vision Transformers
Official Repository of <a href="https://arxiv.org/abs/2206.08477"> ''Backdoor Attacks on Vision Transformers''</a>.

![transformer_teaser](https://user-images.githubusercontent.com/32045261/177569095-a0d2585e-7511-4e0f-8d87-8680599f0ede.jpg)

## Requirements

- Python >= 3.7.6
- PyTorch >= 1.4
- torchvision >= 0.5.0
- timm==0.3.2

## Dataset creation
We follow the same steps as <a href="https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks"> ''Hidden Trigger Backdoor Attacks''</a> for dataset preparations. We repeat the instructions here for convenience.

```python
python create_imagenet_filelist.py cfg/dataset.cfg
```

+ Change ImageNet data source in dataset.cfg

+ This script partitions the ImageNet train and val data into poison generation, finetune and val to run HTBA attack. Change this for your specific needs.



## Configuration file

+ Please create a separate configuration file for each experiment.
+ One example is cfg/singlesource_singletarget_1000class_finetune_deit_base/experiment_0001_base.cfg. Create a copy and make desired changes.
+ The configuration file makes it easy to control all parameters (e.g. poison injection rate, epsilon, patch_size, trigger_ID)

## Poison generation
+ First create directory data/transformer/<EXPERIMENT_ID> and a file in it named source_wnid_list.txt which will contain all the wnids of the source categories for the experiment.
```python
python generate_poison_transformer.py cfg/singlesource_singletarget_1000class_finetune_deit_base/experiment_0001_base.cfg
```

## Finetune
```python
python finetune_transformer.py cfg/singlesource_singletarget_1000class_finetune_deit_base/experiment_0001_base.cfg
```

## Test-time defense
```python
python test_time_defense.py cfg/singlesource_singletarget_1000class_finetune_deit_base/experiment_0001_base.cfg
```

## Data

+ We have provided the triggers used in our experiments in data/triggers
+ To reproduce our experiments please use the correct poison injection rates. There might be some variation in numbers depending on the randomness of the ImageNet data split.


## License

This project is under the MIT license.


## Citation
Please cite us using:
```bib
@article{subramanya2022backdoor,
  title={Backdoor Attacks on Vision Transformers},
  author={Subramanya, Akshayvarun and Saha, Aniruddha and Koohpayegani, Soroush Abbasi and Tejankar, Ajinkya and Pirsiavash, Hamed},
  journal={arXiv preprint arXiv:2206.08477},
  year={2022}
}
```
