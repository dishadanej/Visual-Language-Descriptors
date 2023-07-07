from load import *
import torchmetrics
from tqdm import tqdm


seed_everything(hparams['seed'])

bs = hparams['batch_size']

#########################################
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from os import listdir
list_image_path = []
list_txt = []

class image_title_dataset(Dataset):
    def __init__(self, list_image_path,list_txt):

        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        title = self.title[idx]
        return image,title

#-------
directory_root = './Data1'
image_classes_folder_list = listdir(f"./Data1")
for  image_classes_folder in image_classes_folder_list:
  image_classes_image_list = listdir(f"{directory_root}/{ image_classes_folder}/")
  for image in image_classes_image_list:
    image_directory = f"{directory_root}/{ image_classes_folder}/{image}"

    list_image_path.append(image_directory)
    list_txt.append(image_classes_folder)
# use your own data

dataset = image_title_dataset(list_image_path,list_txt)
########################################


dataloader = DataLoader(dataset, bs)

##########
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from os import listdir
BATCH_SIZE = 8
EPOCH = 1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32",device=device,jit=False)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# add your own code to track the training progress.
for epoch in range(EPOCH):
  for batch in dataloader :
      optimizer.zero_grad()

      images,texts = batch

      images= images.to(device)
      texts = texts.to(device)

      logits_per_image, logits_per_text = model(images, texts)

      ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

      total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
      total_loss.backward()
      if device == "cpu":
         optimizer.step()
      else :
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)

torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"model_10_d1.pt")

##########



print("Loading model...")

device = torch.device(hparams['device'])
# load model
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)

model.eval()
model.requires_grad_(False)

print("Encoding descriptions...")

description_encodings = compute_description_encodings(model)

label_encodings = compute_label_encodings(model)


print("Evaluating...")
lang_accuracy_metric = torchmetrics.Accuracy().to(device)
lang_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5).to(device)

clip_accuracy_metric = torchmetrics.Accuracy().to(device)
clip_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5).to(device)

for batch_number, batch in enumerate(tqdm(dataloader)):
    images, labels = batch
    
    images = images.to(device)
    labels = labels.to(device)
    
    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)
    
    
    clip_acc = clip_accuracy_metric(image_labels_similarity, labels)
    clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)
    
    
    image_description_similarity = [None]*n_classes
    image_description_similarity_cumulative = [None]*n_classes
    
    for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
        
        
        dot_product_matrix = image_encodings @ v.T
        
        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
        
        
    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        
    
    descr_predictions = cumulative_tensor.argmax(dim=1)
    
    
    lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
    lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)
    
    

print("\n")

accuracy_logs = {}
accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()
accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()
accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

# print the dictionary
print("\n")
for key, value in accuracy_logs.items():
    print(key, value)

