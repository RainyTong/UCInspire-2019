## Week 02 Record

### Monday 7.8

- Extract from .grip files to dump recent temperature and moisture into database. 

- Add two more attributes *starttime* and *endtime* into databse.



### Tuesday 7.9

- Write a filter to dump data in American range. 
- Refactor our code according to base module. 
- Make a pull request for the refactored code.

> Refactoring codes makes me feeling so good. :smiley:



### Wednesday 7.10

- Modify our refactored code, like modularizing the code and write comments and documents for them, to make them more readable and reusable. 
- Testify the execution of crawler, extractor and dumper on both local pc and server. 
- Make another pr.

> :heart_eyes: Up to now, 489 lines of python code have been done. And they are refactored.
>
> Feeling good cause Captain Yicong teaches me lots of "code style".
>
> Having some progress everyday and focusing on one thing every day is sooooo good. :heart:

To do :

- Learn to use AllenNLP and try to find sth interesting from text.



### Thursday 7.11

- This morning, I searched some information about Allennlp's function and got an understanding of how to use it. Then I showed a demo during the group meeting successfully and leader Yicong said it is useful. :wink:

Progress:

- Use Allennlp to extract emotion features from tweets text. 
- Allennlp has pretrained models, input a text and output a list of emotion and a list of probabilities. 
- Crawl 500 records from database and predict emotions from text, create two tables emotions & emo_in_rec in db. 
- Dump records and corresponding emotions into database.



### Friday 7.12

- Today's morning, I showed a demo of what I had done yesterday — extract emotion features from tweets and dump them into database. Very good :smiley:

Progress:

- Use Allennlp to extract Person X's intent, X's reaction and Person Y's reaction from text. 
- Add reactions, intents, and 3 more tables in database and dumpe records into them. 
- Label 300+ text records.



> I have been here for two weeks. 
>
>For the first week, I was working on the temperature pipeline, to crawl temperature data from nota websites and dump the data into database. Specifically, since Tingxuan was taking part in the moisture pipeline, so we worked together to write crawler, extractor and dumper. We processed both historical data and recent data which updates everyday. 
>
>And for the second week, we continued to work on it because the data source of t & m was changing and then we refactored our code and made a pull request. Then I began to take part in the nlp part to precess the text from tweets. I learned how to use Allennlp, a tool for natual language processing and use it to extract emotion features from tweets. Specifically, it can use A to extract the intent of the subject in the sentence and the reactions of both subject and object. Then I crawl tweet records from database and extract these emotion features and then dump these features into database.

