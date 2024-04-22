import json
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from transformers import CLIPProcessor, CLIPModel
import os
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.linear_model import LogisticRegression

#path to json file and folder with images
json_path = os.path.expanduser('~/MLfinal/ML541CLIP/captions_val2017.json')
image_path = os.path.expanduser('~/MLfinal/ML541CLIP/val2017')


with open(json_path, 'r') as f:
    input_data = []
    for line in f:
        obj = json.loads(line)
        input_data.append(obj)


# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu" 

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Define a custom dataset
class image_title_dataset():
    def __init__(self, list_image_path,list_txt):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title  = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title

list_image_path = []
list_txt = []
input_data = input_data[0]
for item in input_data['annotations']:
  id = item['image_id']
  id = str(id).zfill(12)
  img_path = image_path + '/' + id + '.jpg'
  caption = item['caption'][:40]
  list_image_path.append(img_path)
  list_txt.append(caption)
train_list_image_path, test_list_image_path, train_list_txt, test_list_txt = train_test_split(list_image_path, list_txt, test_size=0.2, random_state=42)

# Create train dataset
train_dataset = image_title_dataset(train_list_image_path, train_list_txt)
train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

# Create test dataset
test_dataset = image_title_dataset(test_list_image_path, test_list_txt)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
  print('CPU')

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset


# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()

        images,texts = batch 
        
        images= images.to(device)
        texts = texts.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")


# Evaluate the model now
def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

model.eval()
# Calculate the image features
train_features, train_labels = get_features(train_dataset)
test_features, test_labels = get_features(test_dataset)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")