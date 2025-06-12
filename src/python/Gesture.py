import cv2
import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import src.python.utils as utils
import src.python.const as const

def predict_video(
    video_path: str,
    cnn_checkpoint_dir: str = const.MODELS_PATH,
    rnn_checkpoint_dir: str = const.MODELS_PATH,
    device: str = 'cpu'
) -> str:
    """
    Recebe o caminho de um vídeo, extrai os frames, extrai features pela CNN,
    agrupa numa sequência para a GRU e retorna o nome do gesto predito.
    """

    cnn_model, _ = utils.load_CNN(save_dir=cnn_checkpoint_dir, device=device)
    gru_model, label_map = utils.load_RNN(save_dir=rnn_checkpoint_dir, device=device)
    cnn_model.eval().to(device)
    gru_model.eval().to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if not frames:
        raise RuntimeError(f"Não consegui extrair frames de {video_path}")
    with torch.no_grad():
        tensor_frames = [transform(f) for f in frames]      # lista de [C,H,W]
        batch = torch.stack(tensor_frames, dim=0).to(device) # [T, C, H, W]
        T = batch.size(0)
        feats = cnn_model(batch)                             # [T, D]
        padded = pad_sequence([feats.cpu()], batch_first=True, padding_value=0).to(device)
       
        lengths = torch.tensor([T], dtype=torch.long, device=device)

    
        outputs = gru_model(padded, lengths)                
        pred_idx = outputs.argmax(dim=1).item()
    return label_map[pred_idx]

# Exemplo de uso:
if __name__ == "__main__":
    import const
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_file = const.VIDEOS_PATH / "10CincoSinalizador10-2.mp4"
    gesture = predict_video(
        video_file,
        cnn_checkpoint_dir=const.MODELS_PATH,
        rnn_checkpoint_dir=const.MODELS_PATH,
        device=device
    )
    print(f"Gesto detectado: {gesture}")
