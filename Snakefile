
folds = [0, 1, 2, 3, 4]

ALL = expand("logs/experiments/runs/BaseTransformer/fold{fold}/checkpoints/best.ckpt", fold=folds)

rule all:
    input:
        ALL
        
rule train:
    output:
        "logs/experiments/runs/{name}/fold{fold}/checkpoints/best.ckpt"
    shell:
        "python train.py "
        "trainer.gpus=[1] "
        "name={wildcards.name} "
        "fold={wildcards.fold} "