"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from loguru import logger as loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False


def setup_logger(
    name: str = "scitran",
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_loguru: bool = True
) -> logging.Logger:
    """
    Set up logger with configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        use_loguru: Use loguru if available
        
    Returns:
        Configured logger
    """
    if use_loguru and HAS_LOGURU:
        loguru_logger.remove()
        
        loguru_logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            loguru_logger.add(
                log_file,
                level=level,
                rotation="10 MB",
                retention="1 week"
            )
        
        return LoguruWrapper(loguru_logger)
    
    else:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


def get_logger(name: str = "scitran") -> logging.Logger:
    """Get existing logger or create new one."""
    if HAS_LOGURU:
        return LoguruWrapper(loguru_logger)
    else:
        return logging.getLogger(name)


class LoguruWrapper:
    """Wrapper for loguru to standard logging interface."""
    
    def __init__(self, logger):
        self._logger = logger
    
    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)
