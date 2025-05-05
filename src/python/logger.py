import const
import logging


def log_file(path, namefile):
    logger = logging.getLogger(namefile)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
    
    
loggerCNN = log_file(const.LOG_CNN_PATH,const.LOG_CNN)
loggerRNN = log_file(const.LOG_RNN_PATH,const.LOG_RNN)