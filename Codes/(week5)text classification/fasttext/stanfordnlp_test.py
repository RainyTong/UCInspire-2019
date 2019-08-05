from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

s = "Terrifying!! This man and his son trying to escape a Montana wildfire! They eventually got help and escaped by boat! https://t.co/Ve3A9ndnTZ"

res = nlp.annotate(s,
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
# print(res)
for s in res['sentences']:
    print("%d: '%s': %s %s" % (
        s["index"],
        " ".join([t["word"] for t in s["tokens"]]),
        s["sentimentValue"], s["sentiment"]))