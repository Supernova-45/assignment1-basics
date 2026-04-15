# Improving the Transformer performance
## Experiment 1: tune learning rate

```
"d_model": 512,
"num_heads": 16,
"num_layers": 4,
"d_ff": 1344,
"vocab_size": 10000,
"context_length": 256,
"rope_theta": 10000.0,
"lr": lr,
"lr_min": 1e-4,
"warmup_steps": 500,
"betas": [0.9, 0.999],
"eps": 1e-8,
"weight_decay": 0.01,
"num_steps": 10000,
"batch_size": 128,
"max_grad_norm": 1.0,
```
- Tried learning rates: 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
- Validation TinyStores performance:
```
0.01	1.3318151235580444
0.005	1.369703769683838
0.001	1.3712025880813599
0.0005	1.475642442703247
0.0001	1.6786088943481445
```
- Tried learning rates: 0.007, 0.008, 0.012, 0.015, 0.02

```
0.02	3.075204849243164
0.015	2.9736557006835938
0.012	2.8506641387939453
0.008	1.314741849899292
0.007	1.334202527999878
0.01	1.3318151235580444
0.005	1.369703769683838
0.001	1.3712025880813599
0.0005	1.475642442703247
0.0001	1.6786088943481445
```

- Tried learning rates: 0.0075, 0.0085, 0.009, 0.05, 0.1
```
0.1	3.581024408340454
0.009	2.6330597400665283
0.05	3.3446900844573975
0.0085	2.4599127769470215
0.0075	1.3829820156097412
0.02	3.075204849243164
0.015	2.9736557006835938
0.012	2.8506641387939453
0.008	1.314741849899292
0.007	1.334202527999878
```

## Experiment 2: tune batch size
```
"d_model": 512,
"num_heads": 16,
"num_layers": 4,
"d_ff": 1344,
"vocab_size": 10000,
"context_length": 256,
"rope_theta": 10000.0,
"lr": 0.008,
"lr_min": 1e-4,
"warmup_steps": 500,
"betas": [0.9, 0.999],
"eps": 1e-8,
"weight_decay": 0.01,
"num_steps": 10000,
"batch_size": batch_size,
"max_grad_norm": 1.0,
"device": "cuda",
"architecture": "TransformerLM",
"checkpoint_path": str(DATA_PATH / "checkpoints" / f"bs_{batch_size}.pt"),
```

- Tried batch size of 2^3 to 2^10; 2^10 caused memory issue. 
```
512	1.2345161437988281
256	1.2705230712890625
128	1.3497304916381836
64	2.836747169494629
32	2.8744723796844482
8	3.0191643238067627
16	3.0663623809814453
```

## Experiment 3: Removed RMSNorm
Using a learning rate of 0.005 diverged. Here are the other learning rates and validation loss:
0.001	1.4389164447784424
0.0001	1.8778798580169678
0.00005	1.9859403371810913
0.00001	2.0537354946136475

## Experiment 4: Tried Post-Norm
Of 0.0001, 0.0005, 0.001 -- 0.001 was the optimal learning rate and led to a final validation loss of around 1.35.

## Experiment 5: Remove NoPE
With nope (lr = 0.005, tuned), we get a validation loss of 1.385. With rope (lr = 0.008), we get a validation loss of 1.31.

0.005	1.3851025104522705
0.001	1.4075020551681519
0.0005	1.5003514289855957

## Experiment 7: Compare SiLU and SwiGLU
0.01	1.4567763805389404
0.005	1.5718594789505005
0.001	1.6242971420288086
0.0005	1.7637431621551514
0.0001	1.9963139295578003

# Tuning OWT

1. LR sweep

"d_model": 512,
"num_heads": 16,
"num_layers": 4,
"d_ff": 4 * 512,
"vocab_size": 10000,
"context_length": 256,
"rope_theta": 10000.0,
"lr": lr,
"lr_min": 1e-4,
"warmup_steps": 500,
"betas": [0.9, 0.999],
"eps": 1e-8,
"weight_decay": 0.01,
"num_steps": 10000,
"batch_size": 256,
"max_grad_norm": 1.0,

0.015	3.4413485527038574
0.01	5.174596786499023
0.008	3.430220127105713
0.001	3.537924289703369
0.005	3.4633750915527344