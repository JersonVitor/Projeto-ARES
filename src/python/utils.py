#---------- biblioteca padrão ---------- 
import os
import const
import matplotlib.pyplot as plt

#---------- biblioteca de terceiros ---------- 
import torch
def load_models(save_dir="saved_models", device='cpu'):
    # Carregar mapeamento de labels
    checkpoint = torch.load(os.path.join(save_dir, "models_checkpoint.pth"), map_location=device)
    label_map = checkpoint['label_map']
    
    # Carregar arquivos TorchScript
    cnn_model = torch.jit.load(os.path.join(save_dir, "cnn_model.pt"), map_location=device)
    gru_model = torch.jit.load(os.path.join(save_dir, "gru_model.pt"), map_location=device)
    
    # Carregar estados para treino continuado (opcional)
    cnn_model.load_state_dict(checkpoint['cnn_state_dict'])
    gru_model.load_state_dict(checkpoint['gru_state_dict'])
    
    return cnn_model, gru_model, label_map

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
    
    
    