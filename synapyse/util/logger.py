import logging
import sys

__author__ = 'Douglas Eric Fonseca Rodrigues'


class Logger():
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    WARN = logging.WARN
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    logger = logging.getLogger('synapyse')

    @staticmethod
    def is_debug_enabled():
        return Logger.is_enabled_for(Logger.DEBUG)

    @staticmethod
    def is_enabled_for(level):
        """
        :type level: int
        """
        return Logger.logger.isEnabledFor(level)

    @staticmethod
    def info(*args):
        message = ''

        for v in args:
            message += str(v)

        Logger.logger.info(message)

    @staticmethod
    def debug(*args):
        message = ''

        for v in args:
            message += str(v)

        Logger.logger.debug(message)

    @staticmethod
    def enable_logger(level):
        """
        :type level: int
        """
        Logger.logger.setLevel(level)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        Logger.logger.addHandler(ch)