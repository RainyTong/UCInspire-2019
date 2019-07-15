from allennlp.predictors.predictor import Predictor
import psycopg2

class RctnItntProcessor:

    INTENT_TOKENS = 'xintent_top_k_predicted_tokens'
    INTENT_PROB = 'xintent_top_k_log_probabilities'
    REACTION_X_TOKENS = 'xreact_top_k_predicted_tokens'
    REACTION_X_PROB = 'xreact_top_k_log_probabilities'
    REACTION_Y_TOKENS = 'oreact_top_k_predicted_tokens'
    REACTION_Y_PROB = 'oreact_top_k_log_probabilities'

    TABLE_X_IN_RCD = "reaction_x_in_records"
    TABLE_Y_IN_RCD = "reaction_y_in_records"
    TABLE_INTENT_IN_RCD = "intent_in_records"

    URL_EVENT2MIND = "https://s3-us-west-2.amazonaws.com/allennlp/models/event2mind-2018.10.26.tar.gz"

    def __init__(self):
        # set up connection to databse
        self.conn = psycopg2.connect(dbname="testdb", user="tester", password="testpassword",
                                host="cloudberry05.ics.uci.edu", port="5432")
        # set up emotion predictor
        self.predictor = Predictor.from_path(RctnItntProcessor.URL_EVENT2MIND)

    def start(self):
        # crawl records from database
        record_list = self.record_crawler(record_count=100)

        # dump emotions into database
        self.call_dumper(record_list)

    def record_crawler(self, record_count):
        cur = self.conn.cursor()
        cur.execute("SELECT id, text  from records")
        record_dict = cur.fetchmany(record_count)
        self.conn.commit()
        cur.close()
        return record_dict

    def call_dumper(self, record_list):
        for record in record_list:
            intent, probabilities_intent, reactions_x, probabilities_x, reactions_y, probabilities_y\
                = self.call_predictor(record[1])
            self.insert_one(record[0], intent, probabilities_intent,
                            reactions_x, probabilities_x, reactions_y, probabilities_y)

    def call_predictor(self, record):
        predictions = self.predictor.predict(source=record)

        intent = predictions[RctnItntProcessor.INTENT_TOKENS]
        probabilities_intent = predictions[RctnItntProcessor.INTENT_PROB]

        reactions_x = predictions[RctnItntProcessor.REACTION_X_TOKENS]
        probabilities_x = predictions[RctnItntProcessor.REACTION_X_PROB]

        reactions_y = predictions[RctnItntProcessor.REACTION_Y_TOKENS]
        probabilities_y = predictions[RctnItntProcessor.REACTION_Y_PROB]

        return intent, probabilities_intent, reactions_x, probabilities_x, reactions_y, probabilities_y

    def insert_one(self,rid, intents, probabilities_intent,
                            reactions_x, probabilities_x, reactions_y, probabilities_y):
        for i in range(len(reactions_x)):
            emotion = ''
            for word in reactions_x[i]:
                emotion = emotion + word + ' '
            eid = self.insert_into_reactions(emotion)
            self.insert_into_pairs(rid, eid, probabilities_x[i], RctnItntProcessor.TABLE_X_IN_RCD)

        for i in range(len(reactions_y)):
            emotion = ''
            for word in reactions_y[i]:
                emotion = emotion + word + ' '
            eid = self.insert_into_reactions(emotion)
            self.insert_into_pairs(rid, eid, probabilities_y[i], RctnItntProcessor.TABLE_Y_IN_RCD)

        for i in range(len(intents)):
            intent = ''
            for word in intents[i]:
                intent = intent + word + ' '
            eid = self.insert_into_intents(intent)
            self.insert_into_pairs(rid, eid, probabilities_intent[i], RctnItntProcessor.TABLE_INTENT_IN_RCD)

    def insert_into_reactions(self, e):

        cur = self.conn.cursor()

        try:
            cur.execute("SELECT id from reactions where reaction = (%s)", (e,))
            a = cur.fetchone()
            if a:
                return a[0]
            else:
                cur.execute("select max(id) from reactions")
                eid = cur.fetchone()[0]
                # print(eid)
                if eid:
                    eid += 1

                else:
                    eid = 1
                cur.execute("INSERT INTO reactions(id, reaction) values (%s, %s) returning id",
                            (eid, e,))
        except Exception as err:
            print("error", err)

        self.conn.commit()
        cur.close()
        return eid

    def insert_into_intents(self,i):
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT id from intents where intent = (%s)", (i,))
            a = cur.fetchone()
            if a:
                return a[0]
            else:
                cur.execute("select max(id) from intents")
                eid = cur.fetchone()[0]
                # print(eid)
                if eid:
                    eid += 1

                else:
                    eid = 1
                cur.execute("INSERT INTO intents(id, intent) values (%s, %s) returning id",
                            (eid, i,))
        except Exception as err:
            print("error", err)
        self.conn.commit()
        cur.close()
        return eid

    def insert_into_pairs(self, rid, eid, probability, tablename):
        cur = self.conn.cursor()
        try:
            if tablename == RctnItntProcessor.TABLE_X_IN_RCD:
                cur.execute("INSERT INTO reaction_x_in_records(record_id,reaction_x_id,probability) values (%s, %s, %s)",
                            (rid, eid, probability))
            elif tablename == RctnItntProcessor.TABLE_Y_IN_RCD:
                cur.execute("INSERT INTO reaction_y_in_records(record_id,reaction_y_id,probability) values (%s, %s, %s)",
                        (rid, eid, probability))
            elif tablename == RctnItntProcessor.TABLE_INTENT_IN_RCD:
                cur.execute("INSERT INTO intent_in_records(record_id,intent_id,probability) values (%s, %s, %s)",
                        (rid, eid, probability))
        except Exception as err:
            print("error", err)
        self.conn.commit()
        cur.close()



if __name__ == '__main__':
    emo_ext = RctnItntProcessor()
    emo_ext.start()