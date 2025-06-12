import tempfile
import shutil
from fastapi import UploadFile
from pathlib import Path
from src.python.Gesture import predict_video
import torch

async def prediction(video: UploadFile) -> str:
    # Cria arquivo temporário para o vídeo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        shutil.copyfileobj(video.file, temp)
        temp_path = Path(temp.name)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gesture = predict_video(
            str(temp_path),
            device=device
        )
        return gesture
    finally:
        temp_path.unlink(missing_ok=True)