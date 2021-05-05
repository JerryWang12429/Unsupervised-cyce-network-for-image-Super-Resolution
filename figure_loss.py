import os
if __name__ == "__main__":
    file = open("loss_log.txt", "r+")
    content = file.read(100)
    print(content)
