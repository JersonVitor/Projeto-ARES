import cv2
import const
import torch
import cnn
import numpy as np
from collections import deque
import threading
import torch.nn.functional as F
from torchvision import transforms


SEQUENCE_LENGTH = 10
IMG_SIZE = 112
CONFIDENCE_THRESHOLD = 0.4

class GesturePredictor:
    def __init__(self, cnn_path, gru_path, label_map):
        # Carregar modelos otimizados
        print(">>> cnn_model loaded from:", cnn_path)
        print(">>> gru_model loaded from:", gru_path)
        self.cnn_model = torch.jit.load(cnn_path, map_location='cpu').half().eval()
        self.gru_model = torch.jit.load(gru_path, map_location='cpu').half().eval()
        
        self.label_map = label_map
        self.frame_queue = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_queue = deque(maxlen=3)
        self.lock = threading.Lock()
        
        # Pré-processamento
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),  # Manter 3 canais
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ])
        
        # Thread de processamento
        self.processing_thread = None
        self.current_frame = None
        self.running = True

    def preprocess_frame(self, frame):
        # Converter para RGB e redimensionar
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        # Converter para tensor
        tensor = self.transform(frame).half().unsqueeze(0)
        return tensor.to('cpu')

    def process_frames(self):
        while self.running:
            if self.current_frame is not None:
                with self.lock:
                    frame = self.current_frame.copy()
                    self.current_frame = None
                
                processed_tensor = self.preprocess_frame(frame)
                with torch.no_grad():
                    features = self.cnn_model(processed_tensor)
                
                self.frame_queue.append(features)

    def predict_gesture(self):
        if len(self.frame_queue) < SEQUENCE_LENGTH:
            return "Coletando frames...", 0.0, (0, 0, 0)
        
        features = list(self.frame_queue)
        features_tensor = torch.cat(features).unsqueeze(0)
        
        with torch.no_grad():
            gru_out = self.gru_model(features_tensor, torch.tensor([SEQUENCE_LENGTH]))
            probabilities = F.softmax(gru_out, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
        
        confidence = confidence.item()
        pred_idx = pred_idx.item()
        self.prediction_queue.append(pred_idx)
        
        # Filtro temporal
        final_pred = max(set(self.prediction_queue), key=self.prediction_queue.count)
        
        # Mapear cor
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
        
        return self.label_map[final_pred], confidence, color

    def run(self):
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.start()
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            with self.lock:
                self.current_frame = frame.copy()
            
            # Obter predição
            label, confidence, color = self.predict_gesture()
            cv2.putText(
                img=frame,
                text=f"{label} ({confidence:.2f}) [{len(self.frame_queue)}]",
                org=(20, 50),                            # posição x=20,y=50
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,       # fonte padrão
                fontScale=0.8,                           # escala do texto
                color=color,                             # cor que você já define
                thickness=2                              # espessura da linha
            )
                        
            # Interface
            cv2.putText(frame, f"{label} ({confidence:.2f})", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow('Reconhecimento de Gestos - Frame Completo', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.running = False
        self.processing_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        
dataset = cnn.CNNDataset(
        annotations_file=const.CSV_PATH,
        videosDir=const.VIDEOS_PATH,
        transform= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112,112)),
            transforms.Grayscale(num_output_channels=3),  # Manter 3 canais
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ]))


gesture = GesturePredictor(const.CNN_PATH, const.RNN_PATH, dataset.idx2label)
gesture.run()