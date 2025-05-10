# Projeto ARES

O projeto ARES (Aplicação de Reconhecimento e Ensino de Sinais), tem como o objetivo o reconhecimento de gestos de LIBRAS (Língua Brasileira de Sinais) por meio da utiização de duas Redes Neurais: a MobileNetV2 e a GRU.
A escolha dessas duas redes se deve a compatibilidade da biblioteca pytorch, que permite sua execução tanto em CPU quanto em GPU.
No pipeline do Projeto:
- O MobileNet é responsável pela extração da Matriz de Características de cada frame e reunindo em um único arquivo de amostra,
- A GRU(Gated Recorrent Unit) faz a classificação de gestos, interpretando a sequência de Matrizes geradas, fazendo a análise em sua sequência temporal . 

## 📋 Requisitos

Liste os requisitos para rodar o projeto:

- [ ] Sistema operacional: Ubuntu 20.04, Windows 10, dentre outras que possuem compatibilidade com a linguagem e o CUDA toolkit
- [ ] Linguagem: Python 3.12.8
- [ ] CUDA toolkit versão 12.8
- [ ] Dependências/bibliotecas: pytorch, pandas, numpy, tqdm, opencv-python, InquirerPy, ray 

Você pode instalar os pacotes necessários com:

```bash
pip install -r requirements.txt
````

## 🚀 Como Rodar:
#### Clone o repositório
git clone https://github.com/JersonVitor/Projeto-ARES

#### Baixar base de dados para as Redes Neurais:
[MINDS-libras](https://www.kaggle.com/datasets/j0aopsantos/minds-libras)
Colocar na pasta *videos*

#### Navegue até o diretório
cd Projeto-ARES

#### (Opcional) Ative um ambiente virtual
````python
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
````
#### Instale as dependências
pip install -r requirements.txt

#### Navegue até o diretório python
cd src/python

#### Execute o código principal
python main.py

## 📂Estrutura do Projeto  
  ```plaintext
  📂TI6/  
  ├── src/                     # Diretório principal de código-fonte  
  │   ├── 📂features/            # Matrizes de características extraídas (treinamento)  
  │   ├── 📂features_teste/      # Matrizes de características (teste)  
  │   ├── 📂features_val/        # Matrizes de características (validação)  
  │   └── 📂python/              # Módulos Python principais  
  │       ├── 📐logs/            # Logs de execução ou treinamento  
  │       ├── 📂saved_models/    # Modelos treinados salvos  
  │       ├── test/            # Scripts de teste  
  │       ├── 📷cnn.py           # Módulo para extração de características com MobileNetV2  
  │       ├── ⚙const.py         # Constantes do projeto (como paths e configurações)  
  │       ├── Gesture.py       # Classe ou funções para manipulação de gestos  
  │       ├── logger.py        # Sistema de logging personalizado  
  │       ├── main.py          # Ponto principal de execução do projeto  
  │       ├── rnn.py           # Módulo com a rede GRU para classificação  
  │       └── utils.py         # Funções utilitárias  
  ├── 📼videos/                  # Vídeos de entrada usados para extração de gestos  
  ├── .gitignore               # Arquivo de configuração para ignorar arquivos no Git  
  ├── ⚙requirements.txt         # Lista de dependências do projeto  
  ````


## 📝 Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais informações.












