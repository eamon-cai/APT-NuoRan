from webui import uimain

if __name__ == "__main__":
    import sys
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> - <level>{message}</level>",
               colorize=True)
    uimain()


