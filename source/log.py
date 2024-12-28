import colorful as cf
import logging

_time_string = '[%(asctime)s.%(msecs)03d]'
_message_string = '%(message)s'


class _ConsoleFormatter(logging.Formatter):

    _formats = {
        logging.DEBUG: f'{_time_string} DEBUG: {_message_string}',
        logging.INFO: f'{_time_string} {cf.bold("INFO")}: {_message_string}',
        logging.WARNING: f'{_time_string} {cf.bold_yellow("WARNING")}: {_message_string}',
        logging.ERROR: f'{_time_string} {cf.bold_red("ERROR")}: {_message_string}',
        logging.CRITICAL: f'{_time_string} {cf.bold_red("CRITICAL")}: {_message_string}',
    }

    def format(self, record):
        formatter = logging.Formatter(self._formats[record.levelno], datefmt='%H:%M:%S')
        return formatter.format(record)


class _HtmlFormatter(logging.Formatter):

    _formats = {
        logging.DEBUG: f'{_time_string} DEBUG: {_message_string}',
        logging.INFO: f'{_time_string} <span style="font-weight:bold">INFO</span>: {_message_string}',
        logging.WARNING: f'{_time_string} <span style="font-weight:bold;color:orange">WARNING</span>: {_message_string}',
        logging.ERROR: f'{_time_string} <span style="font-weight:bold;color:red">ERROR</span>: {_message_string}',
        logging.CRITICAL: f'{_time_string} <span style="font-weight:bold;color:red">CRITICAL</span>: {_message_string}',
    }

    def format(self, record):
        formatter = logging.Formatter(self._formats[record.levelno], datefmt="%H:%M:%S")
        return formatter.format(record)


_italic_console_decorator = {'open': '[3m', 'close': '[23m[26m'}
_italic_html_decorator = {'open': '<i>', 'close': '</i>'}


class Recorder(logging.Handler):
    def __init__(self, *args, **kwargs):
        logging.Handler.__init__(self, *args)
        self._records = []

    def emit(self, record: logging.LogRecord):
        self._records.append( {'string': self.format(record)
            .replace(_italic_console_decorator['open'], _italic_html_decorator['open'])
            .replace(
                _italic_console_decorator['close'], _italic_html_decorator['close']
            ), 'level': record.levelno}
        )

    def setLevel(self, log_level: int):
        super(Recorder, self).setLevel(log_level)
        self._records = [r for r in self._records if r['level'] >= log_level]

    @property
    def records(self):
        return [r['string'] for r in self._records]

    @property
    def error_count(self):
        return sum(1 for r in self._records if r['level'] == logging.ERROR)

    @property
    def warning_count(self):
        return sum(1 for r in self._records if r['level'] == logging.WARNING)



_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_ConsoleFormatter())
_logger.addHandler(_stream_handler)
_recorder = Recorder()
_recorder.setFormatter(_HtmlFormatter())
_logger.addHandler(_recorder)


def warning(message: str, file: str = None):
    if file != None:
        message += f' ({italic(file)})'
    _logger.warning(message)


def error(message: str, file: str = None):
    if file != None:
        message += f' ({italic(file)})'
    _logger.error(message)


def info(message: str):
    _logger.info(message)


def italic(string: str):
    return f'{_italic_console_decorator["open"]}{string}{_italic_console_decorator["close"]}'


def set_level(level: str):
    level_map = {
        'critical': logging.CRITICAL,
        'fatal': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'warn': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG,
    }

    level_number = level_map[level]
    _recorder.setLevel(level_number)


def get() -> Recorder:
    return _recorder
