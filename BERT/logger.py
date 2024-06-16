import os
import sys
import logging

_log_format = "%(asctime)s |- %(levelname).1s %(name)s - %(message)s"


def _get_log_level():
    """ 
    从环境变量 'LOG_LEVEL' 中获取日志级别

    Returns:
        str|int: e.g. "INFO", 20, "DEBUG", 10, "ERROR", 40.

    """
    level = os.environ.get("LOG_LEVEL", "INFO")
    try:
        level = int(level)
    except ValueError:
        assert isinstance(level, str)
        level = level.upper()
    return level

# 配置日志格式和级别
logging.basicConfig(
    format=_log_format,
    level=_get_log_level(),
    stream=sys.stdout)


class LogLevel(object):
    """定义日志级别常量"""
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    DETAIL = 5
    NOTSET = 0


def init_logger(logger_name=None,
                log_file=os.environ.get("LOG_FILE", ""),
                log_format=_log_format,
                level=_get_log_level()):
    """
    Args:
        logger_name(str): 日志记录器的名称，默认为 None
        log_file(str): 日志文件的路径，默认为 ""
            如果指定了 log_file,则将日志消息输出到文件中,否则默认从环境变量 `LOG_FILE` 中获取。
        log_format(str): 日志格式，默认为指定的格式。
        level(int|logging.Level): 设置日志级别，默认从环境变量 `LOG_LEVEL` 中获取，未设置时默认使用 INFO 级别。
        :: level
            - CRITICAL    50
            - ERROR	40
            - WARNING	30
            - INFO	20
            - DEBUG	10
            - DETAIL  5
            - NOTSET  0
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if log_file:
        handler = logging.FileHandler(log_file)
        if log_format:
            formatter = logging.Formatter(log_format)
            handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


def _test():
    logger = init_logger("test_logger", "test_file.log",
                         level=_get_log_level())
    logger.info("level: {}".format(os.environ.get("LOG_LEVEL", "INFO")))
    import sys
    logger.info(sys.modules[__name__])
    logger.info(logging.getLoggerClass())
    logger.debug("test DEBUG 10")
    logger.info("test INFO 20")
    logger.warning("test WARNING 30")
    logger.error("test ERROR 40")
    logger.critical("test CRITICAL 50")

    if logger.isEnabledFor(logging.DEBUG):
        logger.warning("debug enabled!")
    if logger.isEnabledFor(LogLevel.DEBUG):
        logger.info("detail enabled")


if __name__ == "__main__":
    _test()
