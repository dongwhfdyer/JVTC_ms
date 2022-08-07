file = open('list_duke_test_new.txt', 'w')
for line in open('list_duke_test.txt'):
    line = line.split()

    name = line[0]
    label = int(line[1])
    cam = int(name.split('c')[1][0])
    frame = int(name.split('f')[1][0:7])

    new_line = '%s %d %d %d\n' % (name, label, cam, frame)
    print(new_line)
    file.write(new_line)

file = open('list_duke_train_new.txt', 'w')
for line in open('list_duke_train.txt'):
    line = line.split()

    name = line[0]
    label = int(line[1])
    cam = int(name.split('c')[1][0])
    frame = int(name.split('f')[1][0:7])

    new_line = '%s %d %d %d\n' % (name, label, cam, frame)
    print(new_line)
    file.write(new_line)
