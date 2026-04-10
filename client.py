# client.py
import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.metrics import f1_score, roc_auc_score

from model import CNN


class CNNClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader, device, num_channels=1, num_classes=10):
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = CNN(input_channels=num_channels, num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        #self.optimizer = optim.SGD(
        #self.model.parameters(),
        #lr=0.01,           
        #momentum=0.9,
        #weight_decay=1e-4)
        

        
        #self.optimizer = optim.SGD(
          #self.model.parameters(),
          #lr=0.05,
          #momentum=0.0,
          #weight_decay=1e-4
#)



    def get_parameters(self, config):    #Sending weights TO the Server
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):   #beginnign of every round
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p, dtype=p.dtype, device=self.device)

    def fit(self, parameters, config):
        """
        Works for:
        - FedAvg (proximal_mu=0)
        - FedProx (proximal_mu>0)
        - FedNova-style (returns num_steps)
        Also supports local_epochs passed from the server.
        """
        self.set_parameters(parameters)

        # Hyperparams from server config
        local_epochs = int(config.get("local_epochs", 1))
        proximal_mu = float(config.get("proximal_mu", 0.0))

        # Save copy of global params for FedProx proximal term
        global_params = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        loss_sum = 0.0
        total_examples = 0
        num_steps = 0  

        for _ in range(local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()  #clears old gradients
                logits = self.model(x)   #forward pass
                loss = self.criterion(logits, y)  #computes crossentropy loss

              
                if proximal_mu > 0.0:
                    prox = 0.0
                    for p, g in zip(self.model.parameters(), global_params):
                        prox += torch.sum((p - g) ** 2)
                    loss = loss + (proximal_mu / 2.0) * prox

                loss.backward()     #backpropogate to compute gradients
                self.optimizer.step()    #updates model weights using Adam

                bs = x.size(0)
                loss_sum += loss.item() * bs
                total_examples += bs
                num_steps += 1            #counts every single batch update that is used for fednova

        train_loss = loss_sum / total_examples if total_examples > 0 else 0.0

        return self.get_parameters({}), int(total_examples), {             #gives to teh server the parameters below
            "train_loss": float(train_loss),
            "num_steps": int(num_steps),          # for FedNova aggregation
            "local_epochs": int(local_epochs),    # just for logging/debug
        }

    def evaluate(self, parameters, config):  #After training, the client tests how well the model works on its local data.
        self.set_parameters(parameters)   #loads the global model received from the server into the client model.

        self.model.eval()
        correct, total = 0, 0
        loss_sum = 0.0

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)  #output
                loss = self.criterion(logits, y) #loss

                probs = F.softmax(logits, dim=1)  #probabilities
                preds = logits.argmax(dim=1)  #predicted class

                bs = x.size(0)
                loss_sum += loss.item() * bs
                total += bs
                correct += (preds == y).sum().item()

                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(y.detach().cpu().numpy())
                all_probs.append(probs.detach().cpu().numpy())

        if total == 0:
            return 0.0, 0, {}

        avg_loss = loss_sum / total
        acc = correct / total
        f1 = f1_score(all_labels, all_preds, average="macro")

        y_prob = np.concatenate(all_probs, axis=0)
        try:
            auc = roc_auc_score(np.array(all_labels), y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0

        return float(avg_loss), int(total), {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "auc": float(auc),
        }
