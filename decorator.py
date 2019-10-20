import time


def get_time(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        print('time cost:', end-start, 's')
        return end - start
    return wrapper


def get_time_par(func):
    def wrapper(name):
        start = time.time()
        print(name)
        func(name)
        end = time.time()
        print('time cost:', end-start, 's')
    return wrapper


def get_time_multi_par(func):
    def wrapper(*arg, **args):  # 元组， 字典
        print(arg, args)
        start = time.time()
        func(arg, args)
        end = time.time()
        print('time cost:', end-start, 's')
    return wrapper


@get_time
def run1():
    for i in range(10000000):
        pass
    return 'run1'


@get_time_par
def run2(name):
    for i in range(100000000):
        pass
    print(name)


@get_time_multi_par
def run3(id, name):
    for i in range(100000000):
        pass
    print(id, name)


if __name__ == '__main__':
    s = run1()
    print('run time:', s)
    run2('python')
    run3(1, 'python')