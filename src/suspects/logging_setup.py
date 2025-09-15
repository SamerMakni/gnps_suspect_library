import logging

def setup_logging(name: str = "gnpssuspects", level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
    )
    log = logging.getLogger(name)
    log.setLevel(level)
    return log
