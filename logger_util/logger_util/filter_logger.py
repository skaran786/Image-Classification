from logger_util.rotating_logger import RotatingFileLogger

class FilterLogger(RotatingFileLogger):

    def __init__(self, log_file, max_size=10 * 1024 * 1024, max_backups=5, level=logging.DEBUG, filter_func=None):
        super().__init__(log_file, max_size, max_backups, level)
        if filter_func:
            self.logger.addFilter(filter_func)  # Add custom filter if provided

    def set_filter(self, filter_func):
        self.logger.addFilter(filter_func)  # Allows setting filter after initialization

def create_logger(log_file, filter_func=None, **kwargs):
    """Factory function to create either RotatingFileLogger or FilterLogger based on filter argument"""
    if filter_func:
        return FilterLogger(log_file, filter_func=filter_func, **kwargs)
    else:
        return RotatingFileLogger(log_file, **kwargs)

# Basic rotating file logger
logger = create_logger('training.log', max_size=5 * 1024 * 1024, max_backups=3)
logger.info('Starting training process')

# Filter logger (example filter function)
def log_filter(record):
    return record.levelname in ['WARNING', 'ERROR', 'CRITICAL']  # Only log warning and above

filtered_logger = create_logger('errors.log', filter_func=log_filter)
filtered_logger.warning('Encountered a potential issue during training')
