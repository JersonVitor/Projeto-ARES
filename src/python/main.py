
#biblioteca padrão
import os
import re
import time
import cnn
import rnn
import logger
#biblioteca de terceiros

import cv2
import torch
import kagglehub
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib as ptl
import matplotlib.pyplot as plt
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
from InquirerPy.validator import NumberValidator
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import confusion_matrix, classification_report

#---------- variáveis Globais ----------
import const

#---------- Configuração paralelismo ----------
BATCH_SIZE = 4
NUM_WORKERS = 0






def config_screen():
    resp = config_template()
    threads_value = 4
    interop_value = 1
    cuda_or_cpu = const.CUDAORCPU_VALUE
    while resp is not None:
        match resp :
            case const.THREADS_VALUE:
                threads_value = inquirer.text(message="Quantas Threads serão?[1-8]: ", validate=NumberValidator()).execute()
            case const.INTEROP_VALUE:
                interop_value = inquirer.text(message="Quantas Threads irão trabalhar paralelamente?[1-2]: ", validate=NumberValidator()).execute()
            case const.CUDAORCPU_VALUE:
                cuda_or_cpu = inquirer.select(
                    message="Escolha entre CPU ou GPU",
                    choices=[
                        Choice(value =const.CPU_VALUE, name =const.CPU_VALUE),
                        Choice(value =const.CUDA_VALUE, name =const.CUDA_VALUE),
                        Choice(value = None,name ="VOLTAR")
                    ]
                ).execute()
            case const.CNN_VALUE:
                print()
            case const.GRU_VALUE:
                print()
        resp = config_template()
    with open(const.CONFIG_PATH,"w") as file:
        file.write("threads:"+threads_value+"\n")
        file.write("interop:"+interop_value+"\n")
        file.write("cuda_or_cpu:"+cuda_or_cpu+"\n")

#def test_screen():
    
#def gru_screen():

#def cnn_screen():


        
def home_template():
    return inquirer.select(
        message="______ PROJETO RECONHECIMENTO DE GESTOS COM MobileNetV2 e GRU _____\n",
        choices=[
            Choice(value=const.CONFIG_VALUE, name="Configurações"),
            Choice(value=const.TEST_VALUE,name ="Teste de modelos"),
            Choice(value=const.GRU_VALUE, name="Treinamento com GRU"),
            Choice(value=const.CNN_VALUE, name="Extraindo Características"),
            Choice(value=None, name="SAIR")
        ],
        default=None,
    ).execute()

def config_template():
    return inquirer.select(
        message="______ Configurações _____\n",
        choices=[
            Choice(value =const.THREADS_VALUE , name ="Alterar número de threads"),
            Choice(value =const.INTEROP_VALUE, name ="Alterar número de interop Threads"),
            Choice(value =const.CUDAORCPU_VALUE, name ="Modo de execução(CPU/CUDA)"),
            Choice(value =const.CNN_VALUE, name ="Carregar modelo CNN"),
            Choice(value =const.GRU_VALUE, name ="Carregar modelo RNN"),
            Choice(value = None,name ="VOLTAR")
        ],
        default=None
    ).execute()

def main():
    op = home_template()
    while op is not None:
        match op :
            case const.CONFIG_VALUE:
                config_screen()
            case const.TEST_VALUE:
                cnn.Teste_CNN_Paralelo_forte()
                cnn.Teste_CNN_Paralelo_Fraco()
            case const.GRU_VALUE:
                rnn.initRNN()
            case const.CNN_VALUE:
                cnn.initCNN()
        op = home_template()

if __name__== '__main__':
    main()