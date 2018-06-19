import time
from contextlib import contextmanager

@contextmanager
def timer(msg):

    begin = time.strftime('%H:%M:%S', time.localtime())
    print(f'[*] {begin} {msg} ...')
    yield
    end = time.strftime('%H:%M:%S', time.localtime())
    print(f'[+] {end} {msg} done')


@contextmanager
def ignore(*exceptions):

    try:
        yield
    except exceptions:
        pass


if __name__ == '__main__':

    import os
    with ignore(OSError):
        os.mkdir('audio1-train')
        os.mkdir('audio2-train')