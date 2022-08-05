file = open('list_market_test_new.txt', 'w')
for line in open('list_market_test.txt'):
    line = line.split()

    name = line[0]
    label = int(line[1])
    cam = int(name.split('c')[1][0])

    frame = name.split('/')[1]
    frame = frame.split('_')[2]
    frame = int(frame)

    new_line = '%s %d %d %d\n' % (name, label, cam, frame)
    print(new_line)
    file.write(new_line)

file = open('list_market_train_new.txt', 'w')
for line in open('list_market_train.txt'):
    line = line.split()

    name = line[0]
    label = int(line[1])
    cam = int(name.split('c')[1][0])

    frame = name.split('/')[1]
    frame = frame.split('_')[2]
    frame = int(frame)

    new_line = '%s %d %d %d\n' % (name, label, cam, frame)
    print(new_line)
    file.write(new_line)
