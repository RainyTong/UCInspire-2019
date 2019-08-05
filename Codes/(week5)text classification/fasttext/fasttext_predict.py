import fasttext
import rootpath
import sys
# from backend.data_preparation.connection import Connection

rootpath.append()

model = fasttext.load_model("model_1.0.bin")

text = sys.argv[1]

result = model.predict(text, k=2)

print(result)


# CALCULATING SCORES:

# conn = Connection()()
# cur = conn.cursor()
# cur.execute("SELECT text, label1  from records where label1 = 1 ")
# text_label1 = cur.fetchmany(468)
# text_label1_test = text_label1[400:]
#
# cur.execute("SELECT text, label1  from records where label1 = 0 ")
# text_label1 = cur.fetchmany(468)
# text_label1_test_0 = text_label1[400:]
#
#
# accu_num_1 = 0
# for i in range(len(text_label1_test)):
#     text = str(text_label1_test[i][0]).replace('\n','')
#
#     text = text.encode('ascii', 'ignore').decode('ascii')
#
#     result = model.predict(text, k=2)
#     if result[0][0] == '__label__1':
#         accu_num_1 += 1
#
# accu_num_0 = 0
# for i in range(len(text_label1_test_0)):
#     text = str(text_label1_test_0[i][0]).replace('\n','')
#     text = text.encode('ascii', 'ignore').decode('ascii')
#     result = model.predict(text, k=2)
#     if result[0][0] == '__label__0':
#         accu_num_0 += 1
#
# print("accu_num_1: ",accu_num_1)
# print("accu_num_0: ", accu_num_0)
# total = accu_num_1+accu_num_0
# print("total: ", total)
# print("accuracy: ", total/136)