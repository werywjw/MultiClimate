import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
        inputs = clip_processor(text=transcript, images=image, return_tensors="pt", padding='max_length', truncation=True, max_length=77)

        inputs = {key: value.squeeze(0) for key, value in inputs.items()}

        return inputs, torch.tensor(label)

def collate_fn(batch):
    text_inputs = {key: [] for key in batch[0][0].keys()}
    image_inputs = []
    labels = []

    for inputs, label in batch:
        for key in inputs.keys():
            text_inputs[key].append(inputs[key])
        image_inputs.append(inputs['pixel_values'])
        labels.append(label)

    image_inputs = torch.stack(image_inputs)

    for key in text_inputs.keys():
        text_inputs[key] = torch.nn.utils.rnn.pad_sequence(text_inputs[key], batch_first=True)

    labels = torch.stack(labels)

    return {**text_inputs, 'pixel_values': image_inputs}, labels

class CLIPMultimodalModel(nn.Module):
    def __init__(self):
        super(CLIPMultimodalModel, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(self.clip_model.config.projection_dim, 3) 

    def forward(self, inputs):
        outputs = self.clip_model(**inputs)
        pooled_output = outputs.image_embeds  
        logits = self.classifier(pooled_output)
        return logits

dataset_root = 'dataset'
test_videos = ['ACCFP', 'CCAH', 'CCSAD', 'CCUIM', 'EIB', 'EWCC', 'GGCC', 'SCCC', 'TICC', 'WICC']
val_videos = ['CCGFS', 'CCIAP', 'CICC', 'EFCC', 'FIJI', 'HCCAB', 'HRDCC', 'HUSNS', 'MACC', 'SAPFS']
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

train_dataset = MultimodalDataset(train_videos, dataset_root)
val_dataset = MultimodalDataset(val_videos, dataset_root)
test_dataset = MultimodalDataset(test_videos, dataset_root)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

model = CLIPMultimodalModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(3):  
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        logits = model(inputs)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} - Training Loss: {loss.item()}")

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

torch.save(model.state_dict(), 'best_CLIP.pth')

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