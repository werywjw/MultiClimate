import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BlipModel, AutoProcessor
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

class MultimodalDataset(Dataset):
    def __init__(self, videos, dataset_root):
        self.data = []
        self.dataset_root = dataset_root

        for video in videos:
            csv_path = os.path.join(dataset_root, video, f"{video}.csv")
            data = pd.read_csv(csv_path, header=None, skiprows=1, names=['label', 'text'])

            image_folder = os.path.join(dataset_root, video, f"{video}_frames")

            for index, row in data.iterrows():
                label = row['label']
                transcript = row['text']
                image_path = os.path.join(image_folder, f"{video}-{index + 1:03d}.jpg")

                if os.path.exists(image_path):
                    self.data.append((transcript, image_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transcript, image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        image = image_transform(image)
        return transcript, image, torch.tensor(label)

def collate_fn(batch, max_length=512):
    texts = [item[0] for item in batch]
    images = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length 
    )
    return inputs, labels

class BLIPMultimodalClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(BLIPMultimodalClassificationModel, self).__init__()
        self.blip_model = blip_model

        dummy_inputs = {
            "pixel_values": torch.zeros(1, 3, 224, 224), 
            "input_ids": torch.zeros(1, 512, dtype=torch.long), 
            "attention_mask": torch.ones(1, 512, dtype=torch.long) 
        }

        with torch.no_grad():
            dummy_output = self.blip_model(**dummy_inputs)

        pooled_output_size = dummy_output.image_embeds.shape[-1]  

        self.classifier = nn.Linear(pooled_output_size, num_classes)

    def forward(self, inputs):
        outputs = self.blip_model(**inputs)
        pooled_output = outputs.image_embeds  
        logits = self.classifier(pooled_output)
        return logits

dataset_root = 'dataset'
train_videos = [
    'ACCC', 'AIAQ', 'AIDT', 'AMCC', 'BDCC', 'BECCC', 'BWFF', 'CBAQC', 'CCBN', 'CCBNN',
    'CCCBL', 'CCCP', 'CCCS', 'CCD', 'CCFS', 'CCFWW', 'CCH', 'CCHES', 'CCIAA', 'CCIAH', 'CCICD',
    'CCIS', 'CCISL', 'CCMA', 'CCSC', 'CCTA', 'CCTP', 'CCWC', 'CCWQ', 'CESS', 'COP',
    'CPCC', 'CTCM', 'DACC', 'DFCC', 'DPIC', 'DTECC', 'ECCDS', 'FCC', 'FLW', 'FTACC',
    'HCCAE', 'HCCAW', 'HCCIG', 'HCI', 'HDWC', 'HHVBD', 'HSHWA', 'HSPW', 'IMRF', 'INCAS',
    'MICC', 'NASA', 'OCCC', 'PCOCC', 'PWCCA', 'RAGG', 'RASCC', 'RCCCS', 'RCCS', 'RHTCC',
    'RPDCC', 'SDDA', 'SLCCA', 'SSTCC', 'TCBCC', 'TECCC', 'TIOCC', 'TIYH', 'TTFCC',
    'TUCC', 'UKCC', 'VFVCC', 'VPCC', 'WCCA', 'WFHSW', 'WICCE', 'WISE', 'WTCC', 'YPTL'
]
val_videos = ['CCGFS', 'CCIAP', 'CICC', 'EFCC', 'FIJI', 'HCCAB', 'HRDCC', 'HUSNS', 'MACC', 'SAPFS']
test_videos = ['ACCFP', 'CCAH', 'CCSAD', 'CCUIM', 'EIB', 'EWCC', 'GGCC', 'SCCC', 'TICC', 'WICC']

train_dataset = MultimodalDataset(train_videos, dataset_root)
val_dataset = MultimodalDataset(val_videos, dataset_root)
test_dataset = MultimodalDataset(test_videos, dataset_root)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: collate_fn(x, max_length=512))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: collate_fn(x, max_length=512))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: collate_fn(x, max_length=512))

model = BLIPMultimodalClassificationModel(num_classes=3)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=2e-5)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(3):  
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        logits = model(inputs)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1} - Training Loss: {total_loss / len(train_loader)}")

    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
    
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    print(f"Validation Accuracy: {val_accuracy}, F1 Score: {val_f1}")

torch.save(model.state_dict(), 'best_BLIP.pth')

model.eval()
test_labels = []
test_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average='weighted')
print(f"Test Accuracy: {test_accuracy}, F1 Score: {test_f1}")