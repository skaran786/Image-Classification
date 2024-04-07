import torch
from torch import nn
from pytorch_custom.pytorch_custom.pytorch_base import BaseModel

class MNISTClassifier(BaseModel):

  def __init__(self, input_size=784, hidden_size=128, output_size=10):
    super(MNISTClassifier, self).__init__(input_size, hidden_size, output_size)
    # Define model layers specific to MNIST classification
    self.linear1 = nn.Linear(28 * 28, hidden_size)  # Assuming flattened 28x28 image

  def forward(self, x):
    # Flatten the input image (if not already done)
    x = x.view(x.size(0), -1)
    x = super().forward(x)
    return x

  def train_model(self, train_loader, validation_loader, optimizer, criterion, epochs, patience=5):
    self.train()  # Set model to training mode
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(epochs):
      # Training loop
      for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = self(images)
        loss = criterion(outputs, labels)  # Use provided criterion function

        # Backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
          print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

      # Validation loop (after each training epoch)
      val_loss = self.evaluate_model(validation_loader, criterion)
      print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}')

      # Early stopping logic
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_since_improvement = 0
        # Save the best model (optional)
        # torch.save(self.state_dict(), 'best_model.pt')
      else:
        epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
          print('Early stopping triggered')
          break

  def evaluate_model(self, data_loader, criterion):
    self.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for validation
      losses = []
      for images, labels in data_loader:
        outputs = self(images)
        batch_loss = criterion(outputs, labels)
        losses.append(batch_loss.item())
      return sum(losses) / len(data_loader)

