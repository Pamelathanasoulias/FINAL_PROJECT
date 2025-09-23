"""Loguru setup writing ERROR & logs to logs/logfile.log."""


from loguru import logger

# LOGGING CONFIGURATION
LOG_FILE_PATH = "logfile.log"


# ROTATION = CREATE A NEW FILE WHEN SIZE REACHES 1 MB
# RETENTION = KEEP LOGS FOR 10 DAYS
# LEVEL = ONLY LOG ERRORS OR HIGHER | ERROR, CRITICAL

logger.add(LOG_FILE_PATH,rotation='1 MB',retention="10 days",level="ERROR")
