#---------- biblioteca padrão ---------- 
import re
import os
import time
import utils
from logger import loggerCNN
#---------- biblioteca de terceiros ---------- 
import cv2
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

#---------- variáveis Globais ----------
import const


class CNNDataset(Dataset):
    def __init__(self, annotations_file, videosDir, transform=None, target_transform=None):
        self.annotations_file = annotations_file
        self.labels = self.getLabels()
        classes_em_ordem = list(dict.fromkeys(self.labels["class"]))
        self.label2idx = {classe: i for i, classe in enumerate(classes_em_ordem)}
        self.idx2label = {i: classe for classe, i in self.label2idx.items()}
        self.videos_name = self.getVideosName()
        self.videosDir = videosDir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video = self.extractFrames(self.videosDir + self.videos_name.iloc[idx,0])
        label = self.label2idx[self.labels.iloc[idx,0]]
        video_name = self.videos_name.iloc[idx,0]
        frames_t = [self.transform(frame) for frame in video]  # cada frame→Tensor[C,H,W]
        videoT = torch.stack(frames_t, dim=0)
        if self.target_transform:
            label = self.target_transform(label)
        return videoT, label, video_name

    def getLabels(self):
        labels = pd.read_csv(self.annotations_file)
        return labels[["class"]]

    def getVideosName(self):
        videosName = pd.read_csv(self.annotations_file)
        return videosName[["video_name"]]

    def idxToLabel(self,idx):
        return self.idx2label[idx]

    def extractFrames(self,filepath):
        video = cv2.VideoCapture(filepath)
        frames = []
        while video.isOpened():
            sucess, frame = video.read()
            if not sucess:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        video.release()
        return frames

    def collate_fn(batch):
        sequences,labels, video_name = zip(*batch)
        lengths = [seq.shape[0] for seq in sequences]
        padded_sequences = pad_sequence(sequences=sequences,batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return padded_sequences,labels,video_name,lengths


class CNNMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Carrega MobileNetV2 pré‑treinado
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # Congela todos os parâmetros
        for p in backbone.parameters():
            p.requires_grad = False
          # 3) Extrai apenas as camadas convolucionais
        #    backbone.features é um nn.Sequential que vai até antes do classifier
        self.features = backbone.features

        # 4) Define um pool adaptativo para reduzir [B,1280,H,W] → [B,1280,1,1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)           
        x = self.pool(x)              
        x = x.view(x.size(0), -1)       
        return x
    
    
def initCNN():
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CNNDataset(
        annotations_file=const.CSV_PATH,
        videosDir=const.VIDEOS_PATH,
        transform= transform
    )
    loader = DataLoader(
        dataset,
        batch_size=const.BATCH_SIZE,
        num_workers=const.NUM_WORKERS,    
        pin_memory=const.PIN_MEMORY,    
        collate_fn=CNNDataset.collate_fn
    )
    if const.CUDA and torch.cuda.is_available():
        device = torch.device(const.CUDA_VALUE)
    else: 
        if const.CUDA: print("Não é possivel trabalhar com CUDA, veja se está instalado as bibliotecas!!")
        device = torch.device(const.CPU_VALUE)
        torch.set_num_threads(const.THREADS)
        try:
            torch.set_num_interop_threads(const.INTEROP)
        except:
            print("Já ativou o interop threads anteriormente!!")
            
    cnn_model = feature_matrix(dataloader=loader,device=device)
    utils.save_CNN(cnn_model=cnn_model, label_map= dataset.idx2label)
    
    
def feature_matrix(dataloader:DataLoader, device, video_id_teste= 3, video_id_val=5):
    extrator = CNNMobileNetV2().to(device=device)
    extrator.eval()
    loggerCNN.info("Iniciando extração de features usando MobileNetV2")
    loggerCNN.info('-' * 50)
    with torch.no_grad():
        for batch_idx,(videos, labels, names,lengths) in enumerate(tqdm(dataloader, desc="Extraindo batches")):
            start_time = time.time()
            try:
                B,T,C,H,W = videos.shape
                loggerCNN.info(f"Batch {batch_idx}: shape vídeos = {videos.shape}")
                flat = videos.view(B*T,C,H,W).to(device)
                feats_flats = extrator(flat)
                D = feats_flats.size(1)
                feats = feats_flats.view(B,T,D).cpu()
                for i in range(B):
                    video_name = names[i].replace(".mp4", "")
                    real_T = lengths[i] 
                    x = re.search('-',video_name)
                    chave = int(video_name[x.end()])
                    out_path = f"{const.FEATURES_PATH}{video_name}.pt"  
                    if chave % video_id_val == 0:
                        out_path = f"{const.FEATURES_VAL_PATH}{video_name}.pt"    
                    elif chave % video_id_teste == 0:
                        out_path = f"{const.FEATURES_TESTE_PATH}{video_name}.pt"
                    torch.save({
                            'features': feats[i],
                            'length': real_T
                        }, out_path)
                # métricas: tempo e throughput
                elapsed = time.time() - start_time
                videos_per_sec = B / elapsed
                frames_per_sec = (B * T) / elapsed
                #TODO: ARRUMAR ESSE LOG PARA FICAR COMO UM CSV
                loggerCNN.info(f"Batch {batch_idx} processado em {elapsed:.2f}s — ")
                loggerCNN.info(f"{videos_per_sec:.1f} vídeos/s, {frames_per_sec:.1f} frames/s")
            except Exception as e:
                loggerCNN.error(f"Erro no batch {batch_idx}: {e}", exc_info=True)
            
        to_csv(const.FEATURES_PATH,const.FEATURES_CSV_PATH)
        to_csv(const.FEATURES_TESTE_PATH,const.FEATURES_CSV_TEST_PATH)
        to_csv(const.FEATURES_VAL_PATH,const.FEATURES_CSV_VAL_PATH)
    return extrator
            
def to_csv(path_features, csv_path):
    name_features = os.listdir(path_features)
    class_name = []
    for cl in name_features:
        pos = re.search('Sinalizador',cl) 
        class_name.append(cl[2:pos.start()])
        
    df = pd.DataFrame({
        "video_name": name_features,
        "class": class_name,
    })
    df.to_csv(csv_path,index=False)   