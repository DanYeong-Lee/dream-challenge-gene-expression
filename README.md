<div align="center">

# dream-challenge-hydra

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description
- train
  - Train with training data
  - Validate with validation data (validation data is different according to the 'fold' argument)
  - Log to wandb
  - Save best checkpoint
  - When training ends, test with HQ_testdata & log to wandb
- test
  - Test with HQ_testdata
  - Log to wandb
- predict
  - Make submission.txt file with challenge test data

## How to run

View help to identify all the available parameters
```bash
python train.py --help
```

Train model with chosen model configuration from [configs/model/](configs/model/) (You should write config(.yaml) file first)

```bash
python train.py model=deepfamq_conjoined_adamw
```

You can override any parameter from command line like this (or you can just fix config files)

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64 model.net.conv_kernel_size=15
```

Set 'name' argument to represent the settings you used (ckpt directory & wandb group name are automatically set)
```bash
python train.py model=deepfamq_conjoined_adamw model.net.conv_kernel_size=15 name=deepfamq_conjoined_adamw_conv15
```

Train models with 5-fold CV scheme (You can use snakemake to wrap these runs)
```bash
python train.py model=deepfamq_conjoined_adamw trainer.gpus=[0] fold=0

python train.py model=deepfamq_conjoined_adamw trainer.gpus=[0] fold=1

                              ...
```

Train model with whole training set & validate with HQ_testdata (just set fold=None)
```bash
python train.py model=deepfamq_conjoined_adamw trainer.gpus=[0] fold=None
```

Test model with HQ_testdata (ckpt_path is automatically set according to args 'name' & 'fold')
```bash
python test.py model=deepfamq_conjoined_adamw name=deepfamq_conjoined_adamw_conv15 fold=0
```

Make prediction file to be submitted (ckpt_path is automatically set according to args 'name' & 'fold')
```bash
python predict.py model=deepfamq_conjoined_adamw name=deepfamq_conjoined_adamw_conv15 fold=0
```
