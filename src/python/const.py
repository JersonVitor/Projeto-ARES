#---------- variáveis Globais ----------
from typing import Final

#---------- paths ----------
MODELS_PATH: Final = "src/saved_models/"
VIDEOS_PATH: Final = "src/videos/"
CSV_PATH: Final = "src/videos/annotations.csv"
FEATURES_PATH: Final = "src/features/" 
FEATURES_CSV_PATH: Final = "src/features/annotations.csv"
FEATURES_TESTE_PATH: Final = "src/features_teste/"
FEATURES_CSV_TEST_PATH: Final = "src/features_teste/annotations.csv"
FEATURES_VAL_PATH: Final = "src/features_val/"
FEATURES_CSV_VAL_PATH: Final = "src/features_val/annotations.csv"
CONFIG_PATH:Final ="src/CONFIG.data" 
LOG_CNN: Final = "logCNN.log"
LOG_RNN: Final = "logRNN.log"
LOG_CNN_PATH: Final = "src/python/logs/logCNN.log"
LOG_RNN_PATH: Final = "src/python/logs/logRNN.log"
RNN_GRAPH_PATH: Final = "src/python/logs/graficoRNN.png"
RNN_MATRIX_PATH: Final = "src/python/logs/matrizRNN.png"
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
BATCH_SIZE = 8
NUM_WORKERS = 0
THREADS = 4
INTEROP = 2
PIN_MEMORY = False
CUDA = False
