import logging
import sys

# Logging
# =======
#   - Module level logging and log level settings
# --------------------------------------------------------------
# Log level to use for the module logger (does not apply globally).
LOG_LEVEL = logging.INFO

# Log format to use for the module logger (does not apply globally).
LOG_FORMAT = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(lineno)d:%(message)s")

# Handler for logging outputs.
LOG_HANDLER = logging.StreamHandler(sys.stdout)

# Module logger
LOGGER = logging.getLogger("stroke_ai_persona")

LOGGER.setLevel(LOG_LEVEL)
LOG_HANDLER.setLevel(LOG_LEVEL)
LOG_HANDLER.setFormatter(LOG_FORMAT)
LOGGER.addHandler(LOG_HANDLER)