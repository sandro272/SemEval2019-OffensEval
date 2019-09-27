#! /usr/bin/env python

import pandas as pd
import re

# train = pd.read_csv("../train_data/train_subtask_a", header = 0,  sep = "\t", quoting = 3, encoding = "utf-8")


'''
for content in train["tweet"]:
    if len(content) > 0:
        res1 = re.findall("@USER",content)
        res2 = re.findall("#MAGA",content)
        # print(res2)
'''





'''
# content = "@USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER You are not very smart are you? Why do you think Gen Flynn’s sentencing keeps being rescheduled 1233? URL	@USER @USER Go home you’re drunk!!! @USER #MAGA #Trump2020 👊🇺🇸👊 URL	OFF	TIN	IND"
# print(len(content))
def remove_pattern(input_text,pattern):
    res = re.findall(pattern, input_text)
    for i in res:
        input_text = re.sub(i, '', input_text)
    return input_text

result = remove_pattern(str(train["tweet"]), "@[\w]*").strip()
# result = remove_pattern(content, "@[\w]*").strip()
print(result)
'''


'''
train = pd.read_csv("../train_data/train_subtask_a", header = 0,  sep = "\t", quoting = 3, encoding = "utf-8")
def remove_pattern(input_text, pattern):
    res = re.findall(pattern, input_text)
    for i in res:
        input_text = re.sub(i, '', input_text)
    return input_text


def review_to_wordlist(review_text):

    # review_text = remove_pattern(str(train["tweet"]), "@[\w]*").strip()
    review_text = remove_pattern(review_text, "@[\w]*").strip()
    review_text = review_text.lower()
    # words = stanford_tokenizer(review_text)

    return (review_text)

res = review_to_wordlist(str(train["tweet"]))
print(set(res.split()))
'''

sentences = ["@USER She      you've should ask a few native Americans what their take on this is",
             "@USER @USER  @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER     Go home you’re drunk!!! @USER #MAGA #Trump2020 👊🇺🇸👊, 	@USER @USER @USER She probably did not think it would get this far.",
             "Amazon is investigating Chinese employees who are selling internal data to third-party sellers looking for an edge in the competitive marketplace.",
             '''	'"@USER Someone should''veTaken"" this piece of shit to a volcano.'''
             ]



    # res1 = re.findall("@[\w]*", content)
    # duplicateSpacePattern = re.compile(r'\ +')
    # res1 = re.sub(duplicateSpacePattern, ' ', centent)


    # res1 = re.findall("[^\w+,\'@!#\t\"\-]",content)


for centent in sentences:
    duplicateSpacePattern = re.compile(r'\ +')
    res1 = re.sub(duplicateSpacePattern, ' ', centent)
    res1 = re.sub("(@[\w]*\ )+", "@USER ", res1)

    print(res1)










