import colorful as cf


def warning(message: str):
    print(f'{cf.bold_yellow("WARNING")}: {message}')


def error(message: str):
    print(f'{cf.bold_red("ERROR")}: {message}')


def info(message: str):
    print(message)


def italic(string: str):
    return cf.italic(string)
