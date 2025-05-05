#---------- biblioteca padrão ---------- 
import utils
from logger import loggerRNN
#---------- biblioteca de terceiros ---------- 
import torch
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
DROPOUT = 0.3
NUM_EPOCHS = 80


class RNNDataset(Dataset):
    def __init__(self, annotations_file,featuresDir, transform=None, target_transform=None):
        self.annotations_file = annotations_file
        self.labels = self.getLabels()
        classes_em_ordem = list(dict.fromkeys(self.labels["class"]))
        self.label2idx = {classe: i for i, classe in enumerate(classes_em_ordem)}
        self.idx2label = {i: classe for classe, i in self.label2idx.items()}
        self.features = self.getFeaturesNames()
        self.featuresDir = featuresDir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature_data = self.extractFeature(self.featuresDir + self.features.iloc[idx, 0])
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
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout = 0.3):
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
        #self.dropout = nn.Dropout(dropout)
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
        #last_hidden = self.dropout(last_hidden)

        # Classificador final
        out = self.fc(last_hidden)
        return out


def train_model(device,model, train_loader, val_loader, num_epochs=15):
    loggerRNN.info("Treinamento RNN")
    loggerRNN.info('-' * 50)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # Listas para armazenar métricas
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
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
        epoch_val_loss = val_running_loss / len(val_loader)
        #scheduler.step(epoch_val_loss)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        #TODO: ARRUMAR ESSE LOG TAMBÉM
        loggerRNN.info(f'Epoch {epoch+1}')
        loggerRNN.info(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%')
        loggerRNN.info(f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%')
        loggerRNN.info('-' * 50)
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
        annotations_file=const.FEATURES_CSV_TEST_PATH,
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
    