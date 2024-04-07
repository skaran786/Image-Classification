from datasets import load_dataset
from PIL import Image  # For saving images

# Load the MNIST dataset
mnist = load_dataset("mnist")

# Define the path to save the images
data_folder = "./mnist_images"  # Replace with your desired path

# Function to save image and label
def save_image_label(image, label, index, split="train"):
  image_path = f"{data_folder}/{split}_{index:05d}.png"
  img = Image.fromarray(image.reshape(28, 28))  # Reshape for PIL
  img.save(image_path)

  # Save label (optional)
  # with open(f"{data_folder}/{split}_{index:05d}.txt", "w") as f:
  #   f.write(str(label))

# Iterate through training data and save
for idx, datapoint in enumerate(mnist["train"]):
  save_image_label(datapoint["image"], datapoint["label"], idx)

# Repeat for test data (if needed)
for idx, datapoint in enumerate(mnist["test"]):
  save_image_label(datapoint["image"], datapoint["label"], idx, split="test")

print("MNIST images saved to local folder!")
