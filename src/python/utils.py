#---------- biblioteca padrão ---------- 
import os
import re
import numpy as np
import pandas as pd
import const
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

#---------- biblioteca de terceiros ---------- 
import torch
def load_CNN( save_dir= const.MODELS_PATH, device='cpu'):
    checkpoint = torch.load(os.path.join(save_dir, "models_checkpoint.pth"), map_location=device)
    label_map = checkpoint['label_map']
    cnn_model = torch.jit.load(os.path.join(save_dir, "cnn_model.pt"), map_location=device)
    cnn_model.load_state_dict(checkpoint['cnn_state_dict'])
    return cnn_model, label_map

def load_RNN(save_dir= const.MODELS_PATH, device='cpu'):
    checkpoint = torch.load(os.path.join(save_dir, "models_checkpoint.pth"), map_location=device)
    label_map = checkpoint['label_map']
    gru_model = torch.jit.load(os.path.join(save_dir, "gru_model.pt"), map_location=device)
    gru_model.load_state_dict(checkpoint['gru_state_dict'])
    return gru_model, label_map




def save_CNN(cnn_model, label_map, save_dir= const.MODELS_PATH):
    os.makedirs(save_dir, exist_ok=True)
    
    # Salvar modelos com TorchScript
    cnn_scripted = torch.jit.script(cnn_model.cpu().eval())
    
    # Salvar arquivos
    torch.save({
        'cnn_state_dict': cnn_model.state_dict(),
        'label_map': label_map
    }, os.path.join(save_dir, "cnn_checkpoint.pth"))
    
    cnn_scripted.save(os.path.join(save_dir, "cnn_model.pt"))
    print(f"CNN salvo em: {save_dir}")
    
def save_RNN(gru_model, label_map, save_dir=const.MODELS_PATH):
    os.makedirs(save_dir, exist_ok=True)
    
    # Salvar modelos com TorchScript
    gru_scripted = torch.jit.script(gru_model.cpu().eval())
    
    # Salvar arquivos
    torch.save({
        'gru_state_dict': gru_model.state_dict(),
        'label_map': label_map
    }, os.path.join(save_dir, "models_checkpoint.pth"))
    gru_scripted.save(os.path.join(save_dir, "gru_model.pt"))
    
    print(f"RNN salvo em: {save_dir}")

def plot_graphic(train_losses,val_losses,train_accs,val_accs):
        # Plotar curvas
    plt.figure(figsize=(15, 8))
    
    # Curva de Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Curva de Acurácia
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(const.RNN_GRAPH_PATH)
    plt.show()
    
def to_csv(path_features, csv_path):
    name_features = os.listdir(path_features)
    print(name_features)
    class_name = []
    for cl in name_features:
        pos = re.search('Sinalizador',cl) 
        class_name.append(cl[2:pos.start()])
        
    df = pd.DataFrame({
        "video_name": name_features,
        "class": class_name,
    })
    df.to_csv(csv_path,index=False)   
    

def plot_confusion_matrix(model, dataloader, dataset, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels, lengths in tqdm(dataloader, desc="Gerando matriz de confusão"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences, lengths)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Obter classes e nomes
    classes = sorted(dataset.label2idx.keys())
    class_names = [dataset.idx2label[i] for i in range(len(classes))]
    
    # Calcular matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalizar por linha (por classe real)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plotar
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Matriz de Confusão Normalizada')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(const.RNN_MATRIX_PATH)
    plt.show()
    # Salvar versão não normalizada também
    #plt.figure(figsize=(15, 12))
    #sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
    #            xticklabels=class_names, yticklabels=class_names)
    
    #plt.title('Matriz de Confusão (Contagens Absolutas)')
    #plt.xlabel('Predito')
    #plt.ylabel('Real')
    #plt.xticks(rotation=45, ha='right')
    #plt.yticks(rotation=0)
    #plt.tight_layout()
    #plt.savefig(const.RNN_GRAPH_PATH)
    #plt.show()
