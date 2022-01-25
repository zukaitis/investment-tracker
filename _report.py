import warnings
import colorful as cf

def warn(message: str):
    warnings.warn(f'{cf.bold_yellow("WARNING")}: {message}')

def error(message: str):
    warnings.warn(f'{cf.bold_red("ERROR")}: {message}')

def report(message: str):
    print(message)
