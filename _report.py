import colorful as cf

def warn(message: str):
    print(f'{cf.bold_yellow("WARNING")}: {message}')

def error(message: str):
    print(f'{cf.bold_red("ERROR")}: {message}')

def report(message: str):
    print(message)
