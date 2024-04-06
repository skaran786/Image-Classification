import torch

class BaseModel(torch.nn.Module):

  def __init__(self, input_size, hidden_size, output_size):
    super(BaseModel, self).__init__()
    # Define model layers (replace with your specific architecture)
    self.linear1 = torch.nn.Linear(input_size, hidden_size)
    self.activation = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(hidden_size, output_size)

  def forward(self, x):
    # Pass data through the model layers
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    return x

  def calculate_loss(self, outputs, targets, loss_fn):
    # Calculate loss using the provided loss function
    return loss_fn(outputs, targets)
