import fasttext
import rootpath

rootpath.append()
from backend.data_preparation.connection import Connection


# model = fasttext.train_supervised('train.txt')
# print(model.words)
# print(model.labels)
#
# model.save_model("model_1.0.bin")
# #
# #
# def print_results(N, p, r):
#     print("N\t" + str(N))
#     print("P@{}\t{:.3f}".format(1, p))
#     print("R@{}\t{:.3f}".format(1, r))
#
# print_results(*model.test('test_1.txt'))

# print(model.predict("they are not wildfire they are campus fire", k=2))

trainset = []

with Connection() as conn:
    cur = conn.cursor()
    cur.execute("SELECT text from records where label1 = 0 ")
    text_label1 = cur.fetchmany(468)

    # f = open('data_0.txt','a')
    # print('1')
    for r in text_label1:
        a = str(r[0])

        a = a.encode('ascii','ignore').decode('ascii')

        trainset.append(a.strip().replace('\n','. '))
        # print(a)
        # f.write(a)
        # f.write('\n')


# f = open('data_0.txt','r')
# a = []
#
# with open('data_0.txt', 'r') as f:
#     for line in f:
#         if line == '\n' or line == ' ':
#             continue
#         a.append(line.strip('\n'))

# print(trainset)

for t in trainset:
    print("----------------")
    print(t)

# #
# f = open('test_1.txt','a')
# for r in text_label1_test:
#     a = str(r[0]) + " __label__" + str(r[1])
#     a = a.encode('ascii','ignore').decode('ascii')
#     f.write(a)
#     f.write('\n')

