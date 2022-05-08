
models = ["danq"]
folds = [0, 1, 2, 3, 4]

ALL = expand("logs/experiments/runs/{model}/fold{fold}/checkpoints/best.ckpt", model=models, fold=folds)

rule all:
    input:
        ALL
        
rule train:
    output:
        "logs/experiments/runs/{name}/fold{fold}/checkpoints/best.ckpt"
    shell:
        "python train.py "
        "trainer.gpus=[1] "
        "trainer.max_epochs=50 "
        "model={wildcards.name}.yaml "
        "name={wildcards.name} "
        "fold={wildcards.fold} "