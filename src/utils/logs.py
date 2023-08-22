import coloredlogs
import logging


class ColorLogger:

    @staticmethod
    def get_logger(logger_name):
        logger = logging.getLogger(logger_name)
        coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')
        return logger
