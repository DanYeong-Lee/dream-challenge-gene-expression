
models = ["danq"]
folds = [0, 1, 2, 3, 4]

ALL = expand("logs/experiments/runs/{model}_hidden64pool3/fold{fold}/checkpoints/best.ckpt", model=models, fold=folds)

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
        "model={wildcards.name}.yaml "  # Model name
        
        # Model hyperparameters
        "model.net.pool_size=3 "
        "model.net.fc_hidden_dim=64 "
        
        "name={wildcards.name}_{wildcards.version} "  # ModelCheckpoint output folder name
        "fold={wildcards.fold} "