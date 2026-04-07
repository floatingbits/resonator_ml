import torch
from torch.utils.data import DataLoader

from resonator_ml.machine_learning.training.trainer import LossModule, Predictor




class CompareModels:
    def __init__(self, predictors: dict[str, Predictor], dataloader: DataLoader, loss_module: LossModule):
        self.predictors = predictors
        self.dataloader = dataloader
        self.loss_module = loss_module


    def execute(self):
        counter = 0
        for batch in self.dataloader:
            counter += 1
            print("--------------------- Batch", counter, " - -------------------------------")
            y_pred = {}
            for model_key in self.predictors:
                with torch.no_grad():
                    y_pred[model_key] = self.predictors[model_key].predict(batch)
                print("-----", model_key, "- Batch Loss - -----")
                loss = self.loss_module.compute_loss(y_pred[model_key], batch)
                print("Loss:", loss.item())
            print("------ Samples and predictions - -----")
            for i, input in enumerate(batch[0]) :
                print("Input", input)

                for j, target in enumerate(batch[1][i]):
                    print("Target", target)
                    for model_key in y_pred:
                        print("Pred", model_key, ': ', y_pred[model_key][i][j])



