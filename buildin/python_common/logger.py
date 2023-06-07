import logging.handlers


class Logger(logging.Logger):
    def __init__(self, filename=None):
        super(Logger, self).__init__(self)
        cur_stream = logging.StreamHandler()
        cur_stream.setLevel(logging.DEBUG)
        formater = logging.Formatter(
            "%(asctime)s: %(levelname)s: %(filename)s:%(lineno)d] %(message)s"
        )
        cur_stream.setFormatter(formater)
        self.addHandler(cur_stream)


if __name__ == "__main__":
    pass
