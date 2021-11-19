from threading import Thread

def func():
    print("in thread")

thread = Thread(target=func)
thread.start()
print("in main")
