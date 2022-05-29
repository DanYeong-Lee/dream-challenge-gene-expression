
models = ["deepfamq_conjoined_adamw_gru"]

folds = [0, 1, 2, 3, 4]
#folds = [0, 1, 2, 3, 4]

ALL = expand("logs/experiments/runs/{model}_std_nless/fold{fold}/checkpoints/best.ckpt", model=models, fold=folds)

rule all:
    input:
        ALL
        
rule train:
    output:
        "logs/experiments/runs/{name}_std_nless/fold{fold}/checkpoints/best.ckpt"
    shell:
        "python train.py "
        "trainer.gpus=[0] "
        "trainer.max_epochs=30 "
        
        "model={wildcards.name} "  # Model name
        
        "datamodule=nless "
        
        "name={wildcards.name}_std_nless "  # ModelCheckpoint output folder name
        "fold={wildcards.fold} "