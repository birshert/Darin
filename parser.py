def parse():
    path = "/home/birsert/DARIN/Net/train-1.renju"

    file = open(path, mode='r')

    change = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4,
              'e': 5,
              'f': 6,
              'g': 7,
              'h': 8,
              'j': 9,
              'k': 10,
              'l': 11,
              'm': 12,
              'n': 13,
              'o': 14,
              'p': 15}

    for i in range(10):
        line = file.readline()
        data = line.split()
        if data[0] != 'draw':
            for j in range(1, len(data)):
                data[j] = [change[data[j][:1]] - 1, int(data[j][1:])]
        for elem in data:
            print(elem)


parse()
