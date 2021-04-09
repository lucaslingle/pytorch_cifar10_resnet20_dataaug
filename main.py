import torch as tc
from datasets import CIFAR10HePreprocessing
from classifier import Cifar10ResNet
from runners import Evaluator, Trainer, TrainSpec
import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10 with preprocessing as described in Section 4.2 of He et al., 2015.
training_data = CIFAR10HePreprocessing(root="data", train=True)
test_data = CIFAR10HePreprocessing(root="data", train=False)

batch_size = 128

# Create data loaders.
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# The model from He et al., 2015 for CIFAR-10 uses
# initial_num_filters=16, num_repeats=3, num_stages=3,
# which gives a total of 3*3*2 convolutions in the res blocks, and 20 layers total.
model = Cifar10ResNet(
    img_height=32, img_width=32, img_channels=3,
    initial_num_filters=16, num_repeats=3, num_stages=3, num_classes=10).to(device)
print(model)

try:
    model.load_state_dict(tc.load("model.pth"))
    print('successfully reloaded checkpoint. continuing training...')
except Exception:
    print('no checkpoint found. training from scratch...')

loss_fn = tc.nn.CrossEntropyLoss()

lr_spec = [(0, 0.10), (32000, 0.01), (48000, 0.001), (64000, 0.0)]
train_spec = TrainSpec(max_iters=64000, early_stopping=False, lr_spec=lr_spec)
evaluator = Evaluator(model, test_dataloader)
trainer = Trainer(model, train_dataloader, train_spec, evaluator, verbose=True)

trainer.run(device, loss_fn)
print("Done!")

tc.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    input_example = X[0]
    input_label = y[0]

    x_features = model.visualize(
        tc.unsqueeze(input_example, dim=0))

    y_pred = tc.nn.Softmax()(model(tc.unsqueeze(input_example, dim=0)))

    print('ground truth label: {}'.format(input_label))
    print('predicted label distribution: {}'.format(y_pred))
    print(x_features)

    plt.imshow(np.transpose(input_example, [1,2,0]))
    plt.show()

    break
