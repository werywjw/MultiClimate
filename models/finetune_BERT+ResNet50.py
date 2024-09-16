import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultimodalDataset(Dataset):
    def __init__(self, videos, dataset_root, tokenizer, image_transform):
        self.data = []
        self.tokenizer = tokenizer
        self.image_transform = image_transform

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

        text_inputs = self.tokenizer(transcript, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        text_inputs = {key: value.squeeze(0) for key, value in text_inputs.items()}

        image = Image.open(image_path).convert('RGB')
        image_inputs = self.image_transform(image)

        return text_inputs, image_inputs, torch.tensor(label)

class MultimodalModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased"):
        super(MultimodalModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.resnet = models.resnet50(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)

        self.fusion_layer = nn.Linear(self.bert.config.hidden_size + 512, 512)
        self.classifier = nn.Linear(512, 3)

    def forward(self, text_inputs, image_inputs):
        text_outputs = self.bert(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  

        image_embeddings = self.resnet(image_inputs)

        combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)

        fused_output = self.fusion_layer(combined_embeddings)
        logits = self.classifier(fused_output)
        return logits


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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

train_dataset = MultimodalDataset(train_videos, dataset_root, tokenizer, image_transform)
val_dataset = MultimodalDataset(val_videos, dataset_root, tokenizer, image_transform)
test_dataset = MultimodalDataset(test_videos, dataset_root, tokenizer, image_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = MultimodalModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    for text_inputs, image_inputs, labels in train_loader:
        optimizer.zero_grad()
        
        logits = model(text_inputs, image_inputs)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} - Training Loss: {loss.item()}")

    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for text_inputs, image_inputs, labels in val_loader:
            logits = model(text_inputs, image_inputs)
            preds = torch.argmax(logits, dim=1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
    
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    print(f"Validation Accuracy: {val_accuracy}, F1 Score: {val_f1}")

torch.save(model.state_dict(), 'best_BERTResNet50.pth')

model.eval()
test_labels = []
test_preds = []
with torch.no_grad():
    for text_inputs, image_inputs, labels in test_loader:
        logits = model(text_inputs, image_inputs)
        preds = torch.argmax(logits, dim=1)
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average='weighted')
print(f"Test Accuracy: {test_accuracy}, F1 Score: {test_f1}")