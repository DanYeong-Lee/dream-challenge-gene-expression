
models = ["deepfamq_conjoined_adamw"]

folds = [0]
#folds = [0, 1, 2, 3, 4]

ALL = expand("logs/experiments/runs/{model}_std/fold{fold}/checkpoints/best.ckpt", model=models, fold=folds)

rule all:
    input:
        ALL
        
rule train:
    output:
        "logs/experiments/runs/{name}_{version}/fold{fold}/checkpoints/best.ckpt"
    shell:
        "python train.py "
        "trainer.gpus=[1] "
        "trainer.max_epochs=30 "
        
        "model={wildcards.name} "  # Model name
        
        "name={wildcards.name}_{wildcards.version} "  # ModelCheckpoint output folder name
        "fold={wildcards.fold} "