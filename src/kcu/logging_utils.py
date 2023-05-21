import logging
import logging_loki


def get_loki_logger(url="http://localhost:3100/loki/api/v1/push"):
   logging_loki.emitter.LokiEmitter.level_tag = "level"

   handler = logging_loki.LokiHandler(
      url=url,
      version="1",
   )

   logger = logging.getLogger("my-logger")

   logger.addHandler(handler)
   return logger
