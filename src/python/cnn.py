#---------- biblioteca padrão ---------- 
import re
import os
import time
from pathlib import Path
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
        self.annotations_file = Path(annotations_file)
        self.labels = self.getLabels()
        classes_em_ordem = list(dict.fromkeys(self.labels["class"]))
        self.label2idx = {classe: i for i, classe in enumerate(classes_em_ordem)}
        self.idx2label = {i: classe for classe, i in self.label2idx.items()}
        self.videos_name = self.getVideosName()
        self.videosDir = Path(videosDir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_path = self.videosDir / self.videos_name.iloc[idx,0]
        video = self.extractFrames(video_path)
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
    loggerCNN.info(f"batch_id,runtime,vps,fps")
    tqdm.write("Iniciando extração de features usando MobileNetV2")
    tqdm.write('-' * 50)
    with torch.no_grad():
        for batch_idx,(videos, labels, names,lengths) in enumerate(tqdm(dataloader, desc="Extraindo batches")):
            start_time = time.time()
            try:
                B,T,C,H,W = videos.shape
                tqdm.write(f"Batch {batch_idx}: shape vídeos = {videos.shape}")
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
                loggerCNN.info(f"{batch_idx},{elapsed:.2f},{videos_per_sec:.1f},{frames_per_sec:.1f}")
                tqdm.write(f"Batch {batch_idx} processado em {elapsed:.2f}s — ")
                tqdm.write(f"{videos_per_sec:.1f} vídeos/s, {frames_per_sec:.1f} frames/s")
            except Exception as e:
                loggerCNN.error(f"Erro no batch {batch_idx}: {e}", exc_info=True)
            
        utils.to_csv(const.FEATURES_PATH,const.FEATURES_CSV_PATH)
        utils.to_csv(const.FEATURES_TESTE_PATH,const.FEATURES_CSV_TESTE_PATH)
        utils.to_csv(const.FEATURES_VAL_PATH,const.FEATURES_CSV_VAL_PATH)
    return extrator
            
def run_execution(num_thread, videos_path = const.CSV_PATH ):
    torch.set_num_threads(num_thread)
    device = torch.device('cpu')
    
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CNNDataset(
        annotations_file=videos_path,
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
    start_time = time.time()
    feature_matrix_test(dataloader=loader,device=device)
    return time.time() - start_time 


def Teste_CNN_Paralelo_forte():
    torch.set_num_interop_threads(1)
    results = []
    threads = [1,2,4,8]
    print("Iniciando extração de features usando MobileNetV2 com escalabilidade forte")
    print('-' * 50)
    for n in threads:
        print(f"num_threads = {n}")
        total_time_strong = run_execution(n)
        if n == 1: 
            base_time = total_time_strong
            speedup = 1
            eff = 1
        else: 
            speedup = base_time/ total_time_strong
            eff = (speedup/ n) * 100
        results.append({
        "threads": n,
        "total_time":total_time_strong,
        "speedup": round(speedup, 2),
        "eff": round(eff, 1),
        })
    df = pd.DataFrame(results)
    df.to_csv(const.TEST_ESC_STRONG)
    
    
def Teste_CNN_Paralelo_Fraco():
    results = []
    threads = [1,2,4,8]
    bases = [const.ONE_THREAD_CSV,const.TWO_THREAD_CSV,const.FOUR_THREAD_CSV,const.EIGHT_THREAD_CSV]
    print("Iniciando extração de features usando MobileNetV2 com escalabilidade fraca")
    print('-' * 50)
    for n_thread, csv_path in zip(threads,bases):
        print(f"num_threads = {n_thread} e tamanho da base = {8*n_thread}")
        total_time_strong = run_execution(n_thread,csv_path)
        if n_thread == 1: 
            base_time = total_time_strong
            speedup = 1
            eff = 1
        else: 
            speedup = base_time/ total_time_strong
            eff = speedup * 100
        results.append({
        "threads": n_thread,
        "total_time":total_time_strong,
        "speedup": round(speedup, 2),
        "eff": round(eff, 1),
    })
    df = pd.DataFrame(results)
    df.to_csv(const.TEST_ESC_WEAK)
    
def feature_matrix_test(dataloader:DataLoader, device, video_id_teste= 3, video_id_val=5):
    extrator = CNNMobileNetV2().to(device=device)
    extrator.eval()
    #loggerCNN.info(f"batch_id,runtime,vps,fps")
    #tqdm.write("Iniciando extração de features usando MobileNetV2")
    #tqdm.write('-' * 50)
    with torch.no_grad():
        for batch_idx,(videos, labels, names,lengths) in enumerate(tqdm(dataloader, desc="Extraindo batches")):
            start_time = time.time()
            try:
                B,T,C,H,W = videos.shape
                #tqdm.write(f"Batch {batch_idx}: shape vídeos = {videos.shape}")
                flat = videos.view(B*T,C,H,W).to(device)
                feats_flats = extrator(flat)
                D = feats_flats.size(1)
                feats = feats_flats.view(B,T,D).cpu()
                for i in range(B):
                    video_name = names[i].replace(".mp4", "")
                    real_T = lengths[i] 
                    x = re.search('-',video_name)
                    chave = int(video_name[x.end()])
                    out_path = f"{const.FEATURES_PATH}/{video_name}.pt"  
                    if chave % video_id_val == 0:
                        out_path = f"{const.FEATURES_VAL_PATH}/{video_name}.pt"    
                    elif chave % video_id_teste == 0:
                        out_path = f"{const.FEATURES_TESTE_PATH}/{video_name}.pt"
                    torch.save({
                            'features': feats[i],
                            'length': real_T
                        }, out_path)
                # métricas: tempo e throughput
                elapsed = time.time() - start_time
                videos_per_sec = B / elapsed
                frames_per_sec = (B * T) / elapsed
                #loggerCNN.info(f"{batch_idx},{elapsed:.2f},{videos_per_sec:.1f},{frames_per_sec:.1f}")
                #tqdm.write(f"Batch {batch_idx} processado em {elapsed:.2f}s — ")
                #tqdm.write(f"{videos_per_sec:.1f} vídeos/s, {frames_per_sec:.1f} frames/s")
            except Exception as e:
                loggerCNN.error(f"Erro no batch {batch_idx}: {e}", exc_info=True)
            
        #utils.to_csv(const.FEATURES_PATH,const.FEATURES_CSV_PATH)
        #utils.to_csv(const.FEATURES_TESTE_PATH,const.FEATURES_CSV_TESTE_PATH)
        #utils.to_csv(const.FEATURES_VAL_PATH,const.FEATURES_CSV_VAL_PATH)
    return extrator