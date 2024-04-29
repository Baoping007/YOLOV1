import torch

txty_loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
txty_loss_function = torch.nn.BCELoss(reduction="none")

inputs = torch.tensor(
    [
        [0.2920, 0.5803],
        [0.8159, 0.8591],
    ]
)
target = torch.FloatTensor(
    [
        [0, 1],
        [1, 1],
    ]
)
target = target.sum(dim=-1, keepdim=True)
# print(target)
# loss = txty_loss_function(inputs, target)
# print(loss)
print("{:012}".format(1231231))
