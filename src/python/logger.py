import os
import const
import logging


def log_file(path, namefile):
 # 1) Garante que a pasta exista
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 2) Instância única, limpa handlers antigos
    logger = logging.getLogger(namefile)
    logger.handlers.clear()

    # 3) Aceita tudo ≥ DEBUG, delega filtro aos handlers
    logger.setLevel(logging.DEBUG)

    # 4) Console: só INFO+
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 5) Arquivo: DEBUG+
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)

    # 6) Formato consistente
    fmt = logging.Formatter("")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    # 7) Anexa handlers e desativa propagação
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False

    return logger
    
loggerCNN = log_file(const.LOG_CNN_PATH,const.LOG_CNN)
loggerRNN = log_file(const.LOG_RNN_PATH,const.LOG_RNN)