# Versions

## Resonator
### Model Versions
| Model | hidden layers | activation | audio inputs | control inputs | delay pattern                           |
|-------|---------------| --- |--------------| --- |-----------------------------------------|
| v09   | 2 x 64        | tanh | 8            | 0 (1 dummy) | 1, 2, T-3, T-2, T-1, T, T+1, T+2        |
| v1    | 2 x 64        | tanh | 10           | 0 (1 dummy) | 1, 2,3, T-3, T-2, T-1, T, T+1, T+2, T+3 |
| v1_1  | 2 x 256       | tanh | 10           | 0 (1 dummy) | 1, 2,3, T-3, T-2, T-1, T, T+1, T+2, T+3 |

### Training Parameter Versions

| Parameter Set Version | batch size | epochs | learn rate | loss function |
|-----------------------|------------|-------|-----------|---------------|
| v1                    | 20000      | 200   | 1e-4      | MSE           |
| v1.1                  | 2000       | 2000  | 1e-4      | MSE           |
| v2.1                  | 32         | 200   | 1e-4      | relative_l1   |


### Results

| Model | Training Parameter Version | Snapshot | final epoch loss | Training time | Inference Time per mono sample | intention                                                      | result comment                                                                   | improvement comment                                                     |
|-----|----------------------------|----------|------------------|---------------|---------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| v09 | v1                         | 1        | 3.6e-5           | 266s          | 8.6e-5s        | first test                                                     | no decay when reaching low output                                                | more linearity. Focus loss on smaller powers. Use loss after 5-10 loops |
| v1  | v1.1                       | 2        | 3.6e-5           | 2177s         | 8.6e-5s        | test effect of batch and epoch size                            | decaying with offset, dull sound. no real improvement. End result kind of random | Reset training params.  See above.                                      |
| v1_1 | v1                         | 1        | 5.6e-6           | 390s          | 8.6e-5s        | test effect of throwing neurons/processing power at the system | similar result, even if loss is reduced significantly                            | Use a more sophisticated approach                                       |
| v1  | v1.2                       | 3        | 2.2e-7           | 927s          | 8.6e-5s        | test effect of throwing neurons/processing power at the system | similar result, even if loss is reduced significantly                            | Use a more sophisticated approach                                       |
| v1  | v2.1                       | 5        | 0.49             | 927s          | 8.6e-5s        | test effect of throwing neurons/processing power at the system | similar result, even if loss is reduced significantly                            | Use a more sophisticated approach                                       |
| v1  | v2.2                       | 5        | 0.49             | 927s          | 8.6e-5s        | test effect of throwing neurons/processing power at the system | similar result, even if loss is reduced significantly                            | Use a more sophisticated approach                                       |