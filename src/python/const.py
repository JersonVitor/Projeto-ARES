#---------- variáveis Globais ----------
from typing import Final
from pathlib import Path

# Base do projeto: src/
ROOT: Final = Path(__file__).resolve().parents[1]

# Diretórios
MODELS_PATH: Final = ROOT / "python" / "saved_models"
CNN_PATH: Final = ROOT / "python" / "saved_models/cnn_model.pt"
RNN_PATH: Final = ROOT / "python" / "saved_models/gru_model.pt"
VIDEOS_PATH: Final = ROOT / "videos"
FEATURES_PATH: Final = ROOT / "features"
FEATURES_VAL_PATH: Final = ROOT / "features_val"
FEATURES_TESTE_PATH: Final = ROOT / "features_teste"
LOGS_PATH: Final = ROOT / "python" / "logs"
TEST_PATH: Final = ROOT/"python" / "test/"

# CSVs
ONE_THREAD_CSV:Final = TEST_PATH / "csv_test/80_videos.csv"
TWO_THREAD_CSV:Final = TEST_PATH / "csv_test/160_videos.csv"
FOUR_THREAD_CSV:Final = TEST_PATH / "csv_test/320_videos.csv"
EIGHT_THREAD_CSV:Final = TEST_PATH / "csv_test/640_videos.csv"
CSV_PATH: Final = VIDEOS_PATH / "annotations.csv"
FEATURES_CSV_PATH: Final = FEATURES_PATH / "annotations.csv"
FEATURES_CSV_VAL_PATH: Final = FEATURES_VAL_PATH / "annotations.csv"
FEATURES_CSV_TESTE_PATH: Final = FEATURES_TESTE_PATH / "annotations.csv"
# Config
CONFIG_PATH: Final = ROOT / "python" / "CONFIG.data"

# Logs e gráficos
LOG_CNN: Final = "logCNN.log"
LOG_RNN: Final = "logRNN.log"
LOG_ESC_STRONG: Final = "escalabilidade_forte.csv"
LOG_ESC_WEAK: Final = "escalabilidade_fraca.csv"
LOG_CNN_PATH: Final = LOGS_PATH / LOG_CNN
LOG_RNN_PATH: Final = LOGS_PATH / LOG_RNN
TEST_ESC_STRONG: Final = TEST_PATH / LOG_ESC_STRONG
TEST_ESC_WEAK: Final = TEST_PATH / LOG_ESC_WEAK
RNN_GRAPH_PATH: Final = LOGS_PATH / "graficoRNN.png"
RNN_MATRIX_PATH: Final = LOGS_PATH / "matrizRNN.png"

#---------- strings prompt ----------
CONFIG_VALUE = "config"
TEST_VALUE = "test"
GRU_VALUE = "gru"
CNN_VALUE = "cnn"

THREADS_VALUE = "threads"
INTEROP_VALUE = "interop"
CUDAORCPU_VALUE = "cpuCUDA"
CUDA_VALUE = "cuda"
CPU_VALUE = "cpu"
PLOT_ON_HEAD=True

#---------- Configuraçõa de Redes e Threads ----------
BATCH_SIZE = 2
NUM_WORKERS = 0
THREADS = 4
INTEROP = 2
PIN_MEMORY = False
CUDA = False
