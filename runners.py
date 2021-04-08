import torch as tc
import collections
import numpy as np
TrainSpec = collections.namedtuple('TrainSpec', field_names=['max_iters', 'early_stopping', 'lr_spec'])


class Evaluator:
    def __init__(self, model, test_dataloader):
        self.model = model
        self.test_dataloader = test_dataloader

    def run(self, device, loss_fn):
        num_test_examples = len(self.test_dataloader.dataset)
        self.model.eval()
        test_loss, correct = 0, 0
        with tc.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                test_loss += len(X) * loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(tc.float).sum().item()
        test_loss /= num_test_examples
        correct /= num_test_examples
        return {
            "accuracy": correct,
            "loss": test_loss
        }


class Trainer:
    def __init__(self, model, train_dataloader, train_spec, evaluator, verbose=True):
        self.model = model
        self.train_dataloader = train_dataloader
        self.train_spec = train_spec
        self.evaluator = evaluator
        self.verbose = verbose
        self.global_step = 0 # would be nice if we could checkpoint this like in tensorflow; look into later

    def run(self, device, loss_fn):
        max_iters = self.train_spec.max_iters
        if self.train_spec.early_stopping:
            raise NotImplementedError

        epoch = 1
        optimizer = tc.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        while self.global_step < max_iters:
            if self.verbose:
                print(f"Epoch {epoch}\n-------------------------------")

            for (X, y) in self.train_dataloader:
                if len(X) < self.train_dataloader.batch_size:
                    continue

                X, y = X.to(device), y.to(device)

                # Forward
                logits = self.model(X)
                loss = loss_fn(logits, y)

                # Backprop
                lr = [lr for (start_step, lr) in self.train_spec.lr_spec if start_step <= self.global_step][0]
                optimizer.lr = lr
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update global step
                self.global_step += 1

                if self.global_step % 100 == 0 and self.verbose:
                    loss = loss.item()
                    print(f"loss: {loss:>7f}  [{self.global_step:>5d}/{max_iters:>5d}]")

            # after every epoch, print stats for test set. bad practice, should be validation set. fix later.
            eval_dict = self.evaluator.run(device, loss_fn)
            accuracy = eval_dict['accuracy'] * 100
            test_loss = eval_dict['loss']
            if self.verbose:
                print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")

            epoch += 1
