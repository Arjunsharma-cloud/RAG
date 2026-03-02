import logging
import sys

_loggers = {}

def setup_logger(name: str = "multimodal_rag") -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    _loggers[name] = logger
    return logger

def get_logger(name: str) -> logging.Logger:
    return _loggers.get(name, setup_logger(name))