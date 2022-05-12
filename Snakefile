
models = ["danq_conjoined_reg"]
folds = [0]
#folds = [0, 1, 2, 3, 4]

ALL = expand("logs/experiments/runs/{model}_base/fold{fold}/checkpoints/best.ckpt", model=models, fold=folds)

rule all:
    input:
        ALL
        
rule train:
    output:
        "logs/experiments/runs/{name}_{version}/fold{fold}/checkpoints/best.ckpt"
    shell:
        "python train.py "
        "trainer.gpus=[0] "
        "trainer.max_epochs=50 "
        "+trainer.profiler='pytorch' "
        
        "model={wildcards.name}.yaml "  # Model name
        "model.lr=1e-3 "
        "model.weight_decay=0 "
        
        "name={wildcards.name}_{wildcards.version} "  # ModelCheckpoint output folder name
        "fold={wildcards.fold} "