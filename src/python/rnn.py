#---------- biblioteca padrão ---------- 
import time
import utils
from logger import loggerRNN
#---------- biblioteca de terceiros ---------- 
import torch
import os
from pathlib import Path
import ray
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

#---------- variáveis Globais ----------
import const

INPUT_DIM = 1280 
HIDDEN_DIM = 512
NUM_LAYERS = 2
NUM_CLASSES = 20
DROPOUT = 0.2
NUM_EPOCHS = 80
LEARNING_RATE = 0,001


class RNNDataset(Dataset):
    def __init__(self, annotations_file,featuresDir, transform=None, target_transform=None):
        self.annotations_file = Path(annotations_file)
        self.labels = self.getLabels()
        classes_em_ordem = list(dict.fromkeys(self.labels["class"]))
        self.label2idx = {classe: i for i, classe in enumerate(classes_em_ordem)}
        self.idx2label = {i: classe for classe, i in self.label2idx.items()}
        self.features = self.getFeaturesNames()
        self.featuresDir = Path(featuresDir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature_path = self.featuresDir / self.features.iloc[idx, 0]
        feature_data = self.extractFeature(feature_path)
        features = feature_data['features']
        length = feature_data['length']  # comprimento real (sem padding)
        features = features[:length]  # remove o padding
        label = self.label2idx[self.labels.iloc[idx,0]]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label

    def getFeaturesNames(self):
        df = pd.read_csv(self.annotations_file)
        df["video_name"] = df["video_name"].str.replace("mp4", "pt", regex=False)
        return df[["video_name"]]

    def getLabels(self):
        df = pd.read_csv(self.annotations_file)
        return df[["class"]]
    
    
    def extractFeature(self, path):
        return torch.load(path)

    def rnn_collate_fn(batch):
        sequences, labels = zip(*batch)
        lengths = [seq.shape[0] for seq in sequences]
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return padded_sequences, labels, lengths

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout = 0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Camada GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        # Empacotar sequências para otimização
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        packed_output, hidden = self.gru(packed)
        last_hidden = hidden[-1] 

        # Aplica dropout sobre esse vetor
        last_hidden = self.dropout(last_hidden)

        # Classificador final
        out = self.fc(last_hidden)
        return out


def train_model(device,model, train_loader, val_loader, num_epochs=15):
    runtimeCode = time.time()
    loggerRNN.info(f"epoch,runtime,train_loss,train_acc,val_loss,val_acc")
    tqdm.write("Treinamento RNN")
    tqdm.write('-' * 50)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # Listas para armazenar métricas
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        runtimeEpoch = time.time()
        # Treino
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for sequences, labels, lengths in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Estatísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Métricas de treino
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validação
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Métricas de validação
        runtimeEpoch = time.time() - runtimeEpoch
        epoch_val_loss = val_running_loss / len(val_loader)
        #scheduler.step(epoch_val_loss)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        loggerRNN.info(f'Epoch {epoch+1},{runtimeEpoch}{epoch_train_loss:.4f},{epoch_train_acc:.2f},{epoch_val_loss:.4f},{epoch_val_acc:.2f}')
        tqdm.write(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%')
        tqdm.write(f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%')
        tqdm.write('-' * 50)
    loggerRNN.info(f"tempo total de execução: {time.time() - runtimeCode}ms")
    return model, train_losses, train_accs, val_losses, val_accs


def initRNN():
    dataset = RNNDataset(
        annotations_file = const.FEATURES_CSV_PATH,
        featuresDir= const.FEATURES_PATH
    )
    loader = DataLoader(
        dataset,
        batch_size = const.BATCH_SIZE,
        shuffle = True,
        pin_memory = const.PIN_MEMORY,
        collate_fn = RNNDataset.rnn_collate_fn,
        num_workers = const.NUM_WORKERS
    )
    dataset_val = RNNDataset(
        annotations_file=const.FEATURES_CSV_VAL_PATH,
        featuresDir=const.FEATURES_VAL_PATH
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size =  const.BATCH_SIZE,
        shuffle = True,
        pin_memory = const.PIN_MEMORY,
        collate_fn = RNNDataset.rnn_collate_fn,
        num_workers = const.NUM_WORKERS
    )
    dataset_teste = RNNDataset(
        annotations_file=const.FEATURES_CSV_TESTE_PATH,
        featuresDir=const.FEATURES_TESTE_PATH
    )
    loader_teste = DataLoader(
        dataset_teste,
        batch_size = const.BATCH_SIZE,
        shuffle = True,
        pin_memory = const.PIN_MEMORY,
        collate_fn = RNNDataset.rnn_collate_fn,
        num_workers = const.NUM_WORKERS
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
    gru_model = GRUModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout= DROPOUT
    ).to(device)
    print(device)
    model_trained, train_loss, train_acc, val_loss, val_acc = train_model(device = device,
                                                                          model=gru_model,
                                                                          train_loader=loader,
                                                                          val_loader=loader_val,
                                                                          num_epochs=NUM_EPOCHS)
    utils.save_RNN(gru_model=model_trained, label_map= dataset.idx2label)
    utils.plot_graphic(train_loss,val_loss,train_acc,val_acc)
    utils.plot_confusion_matrix(model=model_trained,dataloader=loader_teste,dataset=dataset_teste,device=device)

    
def initRNNDistribute():
    ray.init(_node_ip_address='192.168.0.114')
    #ray.init(address='auto')  # conecta ao head node iniciado externamente
    # Configurações para Ray
    local_dir_uri = Path("./ray_results").absolute().as_uri()
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            'batch_size': const.BATCH_SIZE,
            'lr': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'input_dim': INPUT_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_workers': const.NUM_WORKERS,
            'pin_memory': const.PIN_MEMORY,
            'num_layers': NUM_LAYERS,
            'num_classes': NUM_CLASSES,
            'dropout': DROPOUT,
            'train_features_csv': const.FEATURES_CSV_PATH,
            'train_features_dir': const.FEATURES_PATH,
            'val_features_csv': const.FEATURES_CSV_VAL_PATH,
            'val_features_dir': const.FEATURES_VAL_PATH
        },
        scaling_config=ScalingConfig(
            num_workers=4,               # ajuste conforme seus nós
            use_gpu=torch.cuda.is_available()
        ),
        run_config=RunConfig(
            name='distributed_rnn_training',
            storage_path=local_dir_uri,
        )
    )

    result = trainer.fit()
    ray.shutdown()
    if hasattr(const, 'PLOT_ON_HEAD') and const.PLOT_ON_HEAD:
        metrics = torch.load('training_metrics.pt')
        utils.plot_graphic(
            metrics['train_loss'], metrics['val_loss'],
            metrics['train_acc'], metrics['val_acc']
        )

# Função de treinamento utilizada por cada worker Ray
def train_loop_per_worker(config):
    # Iniciar contagem de tempo
    runtime_code = time.time()

    # Configurações vindas do config do Ray
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cria datasets e dataloaders
    train_dataset = RNNDataset(
        annotations_file=config["train_features_csv"],
        featuresDir=config["train_features_dir"]
    )
    val_dataset = RNNDataset(
        annotations_file=config["val_features_csv"],
        featuresDir=config["val_features_dir"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=config["pin_memory"],
        collate_fn=RNNDataset.rnn_collate_fn,
        num_workers=config["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=config["pin_memory"],
        collate_fn=RNNDataset.rnn_collate_fn,
        num_workers=config["num_workers"]
    )

    # Instancia modelo
    model = GRUModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    ).to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(config["num_epochs"]):
        runtimeEpoch = time.time()
        # Treino
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for sequences, labels, lengths in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Estatísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Métricas de treino
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validação
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Métricas de validação
        runtimeEpoch = time.time() - runtimeEpoch
        epoch_val_loss = val_running_loss / len(val_loader)
        #scheduler.step(epoch_val_loss)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Report para Ray
        train.report({
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        })
        loggerRNN.info(f'{epoch+1},{runtimeEpoch:.2f},{epoch_train_loss:.4f},{epoch_train_acc:.2f},{epoch_val_loss:.4f},{epoch_val_acc:.2f}')

    # Após treino, salve modelo e métricas
    # Apenas no worker 0 para evitar conflitos
    if train.world_rank() == 0:
        utils.save_RNN(gru_model=model, label_map=train_dataset.idx2label)
        metrics = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
        }
        torch.save(metrics, 'training_metrics.pt')
    print(f"Training finished in {time.time() - runtime_code:.2f}s")


