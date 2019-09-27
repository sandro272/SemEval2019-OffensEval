#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/4 10:29

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re



sentences = ["@USER I got more common sense than all of my followers  :) and it's ",
             "CANT WAIT for f**k the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))"
             ]
def emotion_and_split():

    text_process = TextPreProcessor(

        segmenter="twitter",

        corrector="twitter",

        unpack_hashtags=True,
        unpack_contractions=True,

        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        dicts=[emoticons]
    )

    return text_process

from ekphrasis.classes.spellcorrect import SpellCorrector
def spell_correct(text):
   sp = SpellCorrector(corpus="twitter").correct(text)
   return sp



for sentence in sentences:
    res = spell_correct(sentence)
    res = emotion_and_split().pre_process_doc(res)
    print(res)

    print(str(res))
    # print(type(str(res)))
    res = " ".join(res)

    res = re.sub("@[\w]*", " ",res)
    res = re.sub("[^a-zA-Z0-9\*]", " ", res)
    print(res)
    print(type(res))


