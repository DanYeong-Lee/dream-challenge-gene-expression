
models = ["deepfamq_conjoined"]

folds = [1, 2, 3, 4]
#folds = [0, 1, 2, 3, 4]

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
        "trainer.max_epochs=30 "
        "+trainer.profiler='pytorch' "
        
        "model={wildcards.name} "  # Model name
        "model.lr=1e-3 "
        "model.weight_decay=0 "
        
        "name={wildcards.name} "  # ModelCheckpoint output folder name
        "fold={wildcards.fold} "