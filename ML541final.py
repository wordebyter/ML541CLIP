import json
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import logging
import matplotlib.pyplot as plt
from PIL import Image
import requests
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTFeatureExtractor,
    BertTokenizer,
)
logging.set_verbosity_error()

class CFG:

    max_text_tokens_length = 128
    text_backbone = 'bert-base-uncased'
    image_backbone = 'google/vit-base-patch16-224'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    max_epochs = 75
    max_bad_epochs = 9
    patience = 3
    factor = 0.1

def read_coco_pairs(split='train'):
  if split == 'train':
    path = 'coco_ann2017/annotations/captions_train2017.json'
  else:
    path = 'coco_ann2017/annotations/captions_val2017.json'
  with open(path, 'r') as file:
      data = json.load(file)
  images = list(map(lambda x: (x['id'], x['flickr_url']), data['images']))
  images = {x[0]: x[1] for x in images}
  annotations = list(map(lambda x:(x['image_id'], x['caption']),data['annotations']))
  pairs = list(map(lambda x:{'caption':x[1],'url':images[x[0]]}, annotations))
  return pairs

#memory usage
def print_memory_usage():
    print("Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

class DataSet(torch.utils.data.Dataset):

  def __init__(self, pairs, processor):
    super().__init__()
    self.pairs = pairs
    self.processor = processor
  
  def __getitem__(self, idx):
    try:
      caption = self.pairs[idx]['caption']
      url = self.pairs[idx]['url']
      image = Image.open(requests.get(url, stream=True).raw) 
      encoded_pair = self.processor(text=[caption], images=[image], return_tensors="pt", max_length=CFG.max_text_tokens_length, padding='max_length', truncation=True)
      return encoded_pair
    except:
      return None
  
  def __len__(self):
    return len(self.pairs)

def collate_fn(batch):
  batch = list(filter(lambda x: x is not None, batch))
  return torch.utils.data.dataloader.default_collate(batch)

def train_epoch(model, train_loader, optimizer, epoch, max_epochs):
    model.train()
    nb_batches = len(train_loader)
    tqdm_object = tqdm(train_loader, total=len(train_loader))   
    epoch_loss = 0.0
    for i, batch in enumerate(tqdm_object):
      outputs = model(
          input_ids=batch['input_ids'].squeeze().to(CFG.device),
          attention_mask=batch['attention_mask'].squeeze().to(CFG.device),
          pixel_values=batch['pixel_values'].squeeze().to(CFG.device),
          return_loss=True)
      loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
      loss.backward()
      optimizer.step()
      tqdm_object.set_postfix(
          batch="{}/{}".format(i+1,nb_batches),
          train_loss=loss.item(),
          lr=get_lr(optimizer)
          )
    epoch_loss = epoch_loss / nb_batches
    return epoch_loss

def valid_epoch(model, dev_loader, epoch, max_epochs):
    model.eval()
    nb_batches = len(dev_loader)
    tqdm_object = tqdm(dev_loader, total=len(dev_loader))
    epoch_loss = 0.0   
    for i, batch in enumerate(tqdm_object):
      batch = {key: value.cuda() for key, value in batch.items()}
      outputs = model(
          input_ids=batch['input_ids'].squeeze(),
          attention_mask=batch['attention_mask'].squeeze(),
          pixel_values=batch['pixel_values'].squeeze(),
          return_loss=True)
      loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
      epoch_loss += loss.item()
      tqdm_object.set_postfix(
          batch="{}/{}".format(i+1,nb_batches),
          dev_loss=loss.item(),
          )
    epoch_loss = epoch_loss / nb_batches
    return epoch_loss

def learning_loop(model):
    model.to(CFG.device)
    print_memory_usage()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=CFG.patience, factor=CFG.factor)

    best_dev_score = float('inf')
    train_history = []
    dev_history = []
    nb_bad_epochs = 0

    print("Learning phase")
    print('Used device:', CFG.device)
    print("--------------")
    for epoch in range(1, CFG.max_epochs+1):

        print("Epoch {:03d}/{:03d}".format(epoch, CFG.max_epochs))

        if nb_bad_epochs >= CFG.max_bad_epochs:
            print("Epoch {:03d}/{:03d}: exiting training after too many bad epochs.".format(epoch, CFG.max_epochs))
            torch.save(model.state_dict(), "final.pt")
            break

        else:

            epoch_start_time = time.time()

            epoch_train_loss = train_epoch(model=model, train_loader=train_dataloader, optimizer=optimizer, epoch=epoch, max_epochs=CFG.max_epochs)
            epoch_dev_score = valid_epoch(model=model, dev_loader=val_dataloader, epoch=epoch, max_epochs=CFG.max_epochs)

            duration = time.time() - epoch_start_time

            lr_scheduler.step(epoch_dev_score)

            train_history.append(epoch_train_loss)
            dev_history.append(epoch_dev_score)

            if epoch_dev_score < best_dev_score:
                nb_bad_epochs = 0
                best_dev_score = epoch_dev_score
                torch.save(model.state_dict(), "best.pt")
                print("Finished epoch {:03d}/{:03d} - Train loss: {:.7f} - Valid loss: {:.7f} - SAVED (NEW) BEST MODEL. Duration: {:.3f} s".format(
                epoch, CFG.max_epochs, epoch_train_loss, epoch_dev_score, duration))
            else:
                nb_bad_epochs += 1
                print("Finished epoch {:03d}/{:03d} - Train loss: {:.7f} - Valid loss: {:.7f} - NUMBER OF BAD EPOCH.S: {}. Duration: {:.3f} s".format(
                epoch, CFG.max_epochs, epoch_train_loss, epoch_dev_score, nb_bad_epochs, duration))
    
    history = {'train':train_history,'dev':dev_history}
    return history
  
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def plot_history(history):
    train_history = history['train']
    dev_history = history['dev']
    plt.plot(list(range(1, len(train_history)+1)), train_history, label="train loss")
    plt.plot(list(range(1, len(dev_history)+1)), dev_history, label="dev loss")
    plt.xticks(list(range(1, len(train_history)+1)))
    plt.xlabel("epoch")
    plt.legend()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)

train_pairs = read_coco_pairs(split='train')
val_pairs = read_coco_pairs(split='val')
train_ds = DataSet(pairs=train_pairs, processor=processor)
dev_ds = DataSet(pairs=val_pairs, processor=processor)
train_dataloader = torch.utils.data.DataLoader(train_ds, collate_fn=collate_fn, batch_size=CFG.batch_size)
val_dataloader = torch.utils.data.DataLoader(dev_ds, collate_fn=collate_fn, batch_size=CFG.batch_size)

clip = VisionTextDualEncoderModel.from_vision_text_pretrained(CFG.image_backbone, CFG.text_backbone)

history = learning_loop(clip)
