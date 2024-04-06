import logging


class RotatingFileLogger:

    def __init__(self, log_file, max_size=10 * 1024 * 1024, max_backups=5, level=logging.DEBUG):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)

        # Create a rotating file handler with size and backup configuration
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_size, backupCount=max_backups
        )
        file_handler.setLevel(level)

        # Create a console handler (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set console to log INFO and above

        # Create a formatter for formatting logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            self.logger.addHandler(console_handler)  # Add console handler if not already present

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)