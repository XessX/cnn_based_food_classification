
import data_setup, model_builder, saved_model, train_test
import torch
import os
from torchvision import transforms


EPOCHS = 5
BATCH = 32
HIDDEN_UNIT = 10
LEARNING_RATE = 0.001
TRAIN_DIR = "data/pizza_steak_sushi/train"
TEST_DIR = "data/pizza_steak_sushi/test"

train_dir = TRAIN_DIR
test_dir = TEST_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"

my_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])


my_train_dataloader, my_test_dataloader, my_class_names = data_setup.create_dataloaders(
  train_dir = train_dir,
  test_dir = test_dir,
  transform = my_transform,
  batch_size = BATCH

)

my_model = model_builder.TinyVGG(
  input_shape = 3,
  hidden_units = HIDDEN_UNIT,
  output_shape = len(my_class_names)
).to(device)

my_loss_fn = torch.nn.CrossEntropyLoss()
my_optimizer = torch.optim.Adam(my_model.parameters(), lr=LEARNING_RATE)

train_test.train(
  model = my_model,
  train_dataloader = my_train_dataloader,
  test_dataloader = my_test_dataloader,
  loss_fn = my_loss_fn,
  optimizer = my_optimizer,
  epochs = EPOCHS,
  device = device
)

saved_model.save_model(model = my_model, target_dir = "saved_model", model_name = "my_created_model")
