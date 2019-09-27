#! /usr/bin/env pytho

import re
import pandas as pd
import ekphrasis
from config import TWEMOJI_LIST, LOGOGRAM, TWEMOJI, EMOTICONS_TOKEN
from preprocess_demo import *
from ekphrasis_tool import ekphrasis_config
from stanford_tokenizer_config import *



'''
def remove_pattern(input_text,pattern):
    res = re.findall(pattern, input_text)
    for i in res:
        input_text = re.sub(i, ' ', input_text)
    return input_text
'''

# train = pd.read_csv("./train_data/train_subtask_a", header = 0, sep = "\t", quoting = 3,)

sentences = ["@USER She's should ask a few native Americans what their take on this is",
             "#Liberals f**king up our great country!",
             "@USER @USER Go home you’re drunk!!! @USER #MAGA #Trump2020 👊🇺🇸👊",
             "Amazon is investigating Chinese employees who are selling internal data to third-party sellers looking for an edge in the competitive marketplace.",
             '''	'"@USER Someone should'veTaken"" this piece of shit to a volcano.''',
             "@USER @USER @USER @USER LOL!!!   Throwing the BULLSHIT Flag on such nonsense!!  #PutUpOrShutUp   #Kavanaugh   #MAGA   #CallTheVoteAlready ",
             "We already have enough #LooneyLeft #Liberals fucking up our great country! #Qproofs #TrudeauMustGo",
             "Kind of like when conservatives wanna associate everyone to their left as communist antifa members?",
             "the woman is room Pick one and stick w it Thinking Face",
             "Fuck em bro",
             "LMAO! You are still butt hurt that your loser Hitlary lost.",
             "Almost forgot you was a nigga in ya past life lmao",
             "Lmao fuck u bae BLEW A 20-0 LEAD BLEW A 20-0 LEAD LMAOOOO",
             "Remember the Berlin Wall???",
             "you emem..he must think he is on his way to a brothel",
             "Trump gives a sh*t... and neither do I! LOL URL",
             "I turn the tv off when Juan speaks.in rap beef LMAOOOOOO Nas took THREE MONTHS to respond to Jay and he bodied h iam .",
             "im fucking sad also this gif is art",
             " Only accuse is in the dems playbook!",
             "policies..now he knows best..lol",
             "Fuck u",
             "}:) 3:-) good util 0:-),can't",
             "I CARE THAT HE WAS MURDERED",
             "I'm Assuming I am Not Going to Get an Answer':",
             "OMG if the image reason for Farakans illegal values are because he is black or Muslim",
             "you are a lying corrupt traitor!!!",
             "You can probably i am agine all the SJW snowflakes that do not feel safe working where there might be conservatives lurking in the shadows."

             ]

'''
def review_to_wordlist(review_text):

    review_text = review_text.replace("👊", " Oncoming Fist ")
    review_text = review_text.replace("😂", " Face With Tears of Joy ")
    review_text = review_text.replace(" 🎶 ", " Musical Notes ")
    review_text = review_text.replace("🎶 ", " Musical Notes ")
    review_text = review_text.replace("♥️",  " Heart Suit ")
    review_text = review_text.replace("♥", " Heart Suit ")
    review_text = review_text.replace(" ✔️", " Heavy Check Mark ")
    review_text = review_text.replace("✔", " Heavy Check Mark ")
    review_text = review_text.replace(" 🙉 ", " Hear-No-Evil Monkey ")
    review_text = review_text.replace("🙉", " Hear-No-Evil Monkey ")
    review_text = review_text.replace("🤔", " Thinking Face ")
    review_text = review_text.replace("🤷‍♂", " Man Shrugging ")
    review_text = review_text.replace("️👀	", " Eyes ")
    review_text = review_text.replace("👀", " Eyes ")
    review_text = review_text.replace("🗽", " Statue of Liberty ")
    review_text = review_text.replace("😀", " Grinning Face " )
    review_text = review_text.replace("👌", " OK ")
    review_text = review_text.replace("🙄", " Face With Rolling Eyes ")
    review_text = review_text.replace("💚", " Green Heart ")
    review_text = review_text.replace("🤠", " Cowboy Hat Face ")
    review_text = review_text.replace("👍", " Thumbs Up ")
    review_text = review_text.replace("❤", " Red Heart ")
    review_text = review_text.replace("❤️", " Red Heart ")
    review_text = review_text.replace("*", " Asterisk ")
    review_text = review_text.replace("😌", " Relieved Face ")
    review_text = review_text.replace("😢", " Crying Face ")
    review_text = review_text.replace("💛", " Yellow Heart ")
    review_text = review_text.replace("😑", " Expressionless Face ")
    review_text = review_text.replace("😄", " Grinning Face With Smiling Eyes ")
    review_text = review_text.replace("🤨", " Face With Raised Eyebrow ")
    review_text = review_text.replace("$", " dollar ")
    review_text = review_text.replace("💕", " Two Hearts ")
    review_text = review_text.replace("🙏", " Folded Hands ")
    review_text = review_text.replace("😉", " Winking Face ")
    review_text = review_text.replace("🔥", " Fire ")
    review_text = review_text.replace("👍🏻", " Thumbs Up: Light Skin Tone ")
    review_text = review_text.replace("😅", " Grinning Face With Sweat ")
    review_text = review_text.replace("😍", " Smiling Face With Heart-Eyes ")
    review_text = review_text.replace("😗", " Kissing Face ")
    review_text = review_text.replace("💜", " Purple Heart ")
    review_text = review_text.replace("😭", " Loudly Crying Face ")
    review_text = review_text.replace("🤦🏻‍♀", " Woman Facepalming: Light Skin Tone ")
    review_text = review_text.replace("🤷", " Person Shrugging ")
    review_text = review_text.replace("☝️", " Index Pointing Up ")
    review_text = review_text.replace("☝", " Index Pointing Up ")
    review_text = review_text.replace("🤟", " Love-You Gesture ")
    review_text = review_text.replace("😊", " Smiling Face With Smiling Eyes ")
    review_text = review_text.replace("🌹", " Rose ")
    review_text = review_text.replace("😁", " Beaming Face With Smiling Eyes ")
    review_text = review_text.replace("❣", " Heavy Heart Exclamation ")
    review_text = review_text.replace("👍🏼", " Thumbs Up: Medium-Light Skin Tone ")
    review_text = review_text.replace("👏🏼", " Clapping Hands: Medium-Light Skin Tone ")
    review_text = review_text.replace("🙏🏼", " Folded Hands: Medium-Light Skin Tone ")
    review_text = review_text.replace("✌🏼", " Victory Hand: Medium-Light Skin Tone ")
    review_text = review_text.replace("👮🏻", " Police Officer: Light Skin Tone ")
    review_text = review_text.replace("👩🏻‍✈‍", " Woman Pilot: Light Skin Tone ")
    review_text = review_text.replace("👏🏻‍", " Clapping Hands: Light Skin Tone ")
    review_text = review_text.replace("😛", " Face With Tongue ")
    review_text = review_text.replace("❤️", " Red Heart ")
    review_text = review_text.replace("😆", " Grinning Squinting Face ")
    review_text = review_text.replace("😎", " Smiling Face With Sunglasses ")
    review_text = review_text.replace("🤷‍♀", " Woman Shrugging ")
    review_text = review_text.replace("😚", " Kissing Face With Closed Eyes ")
    review_text = review_text.replace("🙌🏾", " Raising Hands: Medium-Dark Skin Tone ")
    review_text = review_text.replace("😳", " Flushed Face ")
    review_text = review_text.replace("✌️", " Victory Hand ")
    review_text = review_text.replace("🙈", " See-No-Evil Monkey ")
    review_text = review_text.replace("☕️", " Hot Beverage ")
    review_text = review_text.replace("☕", " Hot Beverage ")
    review_text = review_text.replace("🙌🏽", " Raising Hands: Medium Skin Tone ")
    review_text = review_text.replace("💯", " Hundred Points ")
    review_text = review_text.replace("👏🏽", " Clapping Hands: Medium Skin Tone ")
    review_text = review_text.replace("🤣", " Rolling on the Floor Laughing ")
    review_text = review_text.replace("😶", " Face Without Mouth ")
    review_text = review_text.replace("💥", " Collision ")
    review_text = review_text.replace("‼️", " Double Exclamation Mark ")
    review_text = review_text.replace("‼", " Double Exclamation Mark ")
    review_text = review_text.replace("😳", " Flushed Face ")
    review_text = review_text.replace("😘", " Face Blowing a Kiss ")
    review_text = review_text.replace("🎂", " Birthday Cake ")
    review_text = review_text.replace("🙌", " Raising Hands ")
    review_text = review_text.replace("😪", " Sleepy Face ")
    review_text = review_text.replace("🐇", " Rabbit ")
    review_text = review_text.replace("🕳️", " Hole ")
    review_text = review_text.replace("😡", " Pouting Face ")
    review_text = review_text.replace("🙏🏻", " Folded Hands: Light Skin Tone ")
    review_text = review_text.replace("💙", " Blue Heart ")
    review_text = review_text.replace("💝", " Heart With Ribbon ")
    review_text = review_text.replace("😅", " Grinning Face With Sweat ")
    review_text = review_text.replace("🌸", " Cherry Blossom ")
    review_text = review_text.replace("📣", " Megaphone ")
    review_text = review_text.replace("🌪", " Tornado ")
    review_text = review_text.replace("⛏️", " Pick ")
    review_text = review_text.replace("⛏", " Pick ")
    review_text = review_text.replace("👎", " Thumbs Down ")
    review_text = review_text.replace("😩", " Weary Face ")
    review_text = review_text.replace("😣", " Persevering Face ")
    review_text = review_text.replace("🥀", " Wilted Flower ")
    review_text = review_text.replace("🐍", " Snake ")
    review_text = review_text.replace("💞", " Revolving Hearts ")
    review_text = review_text.replace("📱", " Mobile Phone ")
    review_text = review_text.replace("🐕 ", " dog ")
    review_text = review_text.replace("😇", " Smiling Face With Halo ")
    review_text = review_text.replace("😤", " Face With Steam From Nose ")
    review_text = review_text.replace("👊🏽", " Oncoming Fist: Medium Skin Tone ")
    review_text = review_text.replace("⁉️", " Exclamation Question Mark ")
    review_text = review_text.replace("⁉", " Exclamation Question Mark ")
    review_text = review_text.replace("🐨", " Koala ")
    review_text = review_text.replace("🐻", " Bear Face ")
    review_text = review_text.replace("🐨", " Koala ")
    review_text = review_text.replace("🛑", " Stop Sign ")
    review_text = review_text.replace("👉", " Backhand Index Pointing Right ")
    review_text = review_text.replace("👊🏻", " Oncoming Fist: Light Skin Tone ")
    review_text = review_text.replace("🤥", " Lying Face ")
    review_text = review_text.replace("🙋", " Person Raising Hand ")
    review_text = review_text.replace("💋", " Kiss Mark ")
    review_text = review_text.replace("🎁", " Wrapped Gift ")
    review_text = review_text.replace("⭐️", " Star ")
    review_text = review_text.replace("⭐", " Star ")
    review_text = review_text.replace("🤐"," Zipper-Mouth Face ")
    review_text = review_text.replace("❄️"," Snowflake ")
    review_text = review_text.replace("❄", " Snowflake ")
    review_text = review_text.replace("🤢", " Nauseated Face ")
    review_text = review_text.replace("😈", " Smiling Face With Horns ")
    review_text = review_text.replace("🧐", " Face With Monocle ")
    review_text = review_text.replace("🤦🏾‍♀️", " Woman Facepalming: Medium-Dark Skin Tone ")
    review_text = review_text.replace("👋", " Waving Hand ")
    review_text = review_text.replace("🐾", " Paw Prints ")
    review_text = review_text.replace("💀", " Skull")
    review_text = review_text.replace("😋", " Face Savoring Food ")
    review_text = review_text.replace("😢", " Crying Face ")
    review_text = review_text.replace("🤬", " Face With Symbols on Mouth ")
    review_text = review_text.replace("😒", " Unamused Face ")
    review_text = review_text.replace("🤒", " Face With Thermometer ")
    review_text = review_text.replace("💼", " Briefcase ")
    review_text = review_text.replace("🕶", " Sunglasses ")
    review_text = review_text.replace("👢", " Woman’s Boot ")
    review_text = review_text.replace("⚽️", " Soccer Ball ")
    review_text = review_text.replace("⚽", " Soccer Ball ")
    review_text = review_text.replace("🤗", " Hugging Face ")
    review_text = review_text.replace("😩", " Weary Face ")
    review_text = review_text.replace("🤷🏼‍♂", " Man Shrugging: Medium-Light Skin Tone ")
    review_text = review_text.replace("🤷🏼‍♂️", " Man Shrugging: Medium-Light Skin Tone ")
    review_text = review_text.replace("🤦‍♀", " Woman Facepalming")
    review_text = review_text.replace("🤦‍♀️", " Woman Facepalming")
    review_text = review_text.replace("💪🏻", " Flexed Biceps: Light Skin Tone ")
    review_text = review_text.replace("💩", " Pile of Poo")
    review_text = review_text.replace("💉", " Syringe ")
    review_text = review_text.replace("😱"," Face Screaming in Fear ")
    review_text = review_text.replace("😠 "," Angry Face ")
    review_text = review_text.replace("🌍", " Globe Showing Europe-Africa ")
    review_text = review_text.replace("👮‍♀️", " Woman Police Officer ")
    review_text = review_text.replace("👮‍♀", " Woman Police Officer ")
    review_text = review_text.replace("💰", " Money Bag ")
    review_text = review_text.replace("💖", " Sparkling Heart ")
    review_text = review_text.replace("😏", " Smirking Face ")
    review_text = review_text.replace("💁🏽‍♀️", " Woman Tipping Hand: Medium Skin Tone ")
    review_text = review_text.replace("💁🏽‍♀", " Woman Tipping Hand: Medium Skin Tone ")
    review_text = review_text.replace("🌚"," New Moon Face ")
    review_text = review_text.replace("🤯"," Exploding Head ")
    review_text = review_text.replace("you’re", "you are")
    review_text = review_text.replace("You’re", "You are")
    review_text = review_text.replace("You're", "You are")
    review_text = review_text.replace("should've", "should have ")
    review_text = review_text.replace("it’s", "it is")
    review_text = review_text.replace("It’s", "It is")
    review_text = review_text.replace("It's","It is")
    review_text = review_text.replace("doesn’t", "dose not")
    review_text = review_text.replace("What’s", "What is")
    review_text = review_text.replace(" what's", "what is")
    review_text = review_text.replace("you've", "you have")
    review_text = review_text.replace("he’s", "he is")
    review_text = review_text.replace("He’s", "He is")
    review_text = review_text.replace("There's", "There is")
    review_text = review_text.replace("there's", "there is")
    review_text = review_text.replace("women's", "woman is")
    review_text = review_text.replace("men's", "men is")
    review_text = review_text.replace("That's", "That is")
    review_text = review_text.replace("that’s", "that is")
    review_text = review_text.replace("She's", "she is")
    review_text = review_text.replace("she’s", "she is")
    review_text = review_text.replace("didn’t", "did not")
    review_text = review_text.replace("don't", "do not")
    review_text = review_text.replace("don’t", "do not")
    review_text = review_text.replace("Don't", "Do not")
    review_text = review_text.replace("I'll","I will")
    review_text = review_text.replace("I’ll", "I will")
    review_text = review_text.replace("I'd", "I would")
    review_text = review_text.replace("I’ve", "I have")
    review_text = review_text.replace("I’m", "I am")
    review_text = review_text.replace("i’m", "i am")
    review_text = review_text.replace("Let's", "Let is")
    review_text = review_text.replace("won’t", "won not")
    review_text = review_text.replace("can’t", "can not")
    review_text = review_text.replace("hadn't", "had not")
    review_text = review_text.replace("wouldn’t", "would not")
    review_text = review_text.replace("Shouldn't", "Should not")
    review_text = review_text.replace("shouldn't", "should not")
    review_text = review_text.replace("shouldn’t", "should not")
    review_text = review_text.replace("aren't", "are not")
    review_text = review_text.replace("She’s", "She is")
    review_text = review_text.replace("she’s", "she is")
    review_text = review_text.replace("She’ll", "She will")
    review_text = review_text.replace("she’ll", "she will")
    review_text = review_text.replace("Couldn't", "Could not")
    review_text = review_text.replace("isn’t", "is not")


    # review_text = review_text.replace("")


    # review_text = remove_pattern(str(review_text), "@[\w]*")
    review_text = re.sub("@[\w]*", " ", review_text)
    review_text = re.sub("[^a-zA-Z0-9\']", " ", str(review_text))  # 实验3
    # review_text = re.sub("[!?,.]", "", review_text).strip()

    words = review_text.lower()

    return (words)
'''



def review_to_wordlist(review_text):
    repeatedChars = ['.', '?', '!', ',', '"']
    for c in repeatedChars:
        lineSplit = review_text.split(c)
        # print(lineSplit)
        while True:
            try:
                lineSplit.remove('')
            except:
                break
        cSpace = ' ' + c + ' '
        line = cSpace.join(lineSplit)

    emoji_repeatedChars = TWEMOJI_LIST
    for emoji_meta in emoji_repeatedChars:
        emoji_lineSplit = line.split(emoji_meta)
        while True:
            try:
                emoji_lineSplit.remove('')
                emoji_lineSplit.remove(' ')
                emoji_lineSplit.remove('  ')
                emoji_lineSplit = [x for x in emoji_lineSplit if x != '']
            except:
                break
        emoji_cSpace = ' ' + TWEMOJI[emoji_meta][0] + ' '
        review_text = emoji_cSpace.join(emoji_lineSplit)

    review_text = emoji_to_text(review_text)

    review_text = re.sub("(@[\w]*\ )+", " @USER ", review_text)

    duplicateSpacePattern = re.compile(r'\ +')
    review_text = re.sub(duplicateSpacePattern, ' ', review_text).strip()
    # print(review_text)

    string = re.sub("tha+nks ", ' thanks ', review_text)
    string = re.sub("Tha+nks ", ' Thanks ', string)
    string = re.sub("yes+ ", ' yes ', string)
    string = re.sub("Yes+ ", ' Yes ', string)
    string = re.sub("very+ ", ' very ', string)
    string = re.sub("go+d ", ' good ', string)
    string = re.sub("Very+ ", ' Very ', string)
    string = re.sub("why+ ", ' why ', string)
    string = re.sub("wha+t ", ' what ', string)
    string = re.sub("sil+y ", ' silly ', string)
    string = re.sub("hm+ ", ' hmm ', string)
    string = re.sub("no+ ", ' no ', string)
    string = re.sub("sor+y ", ' sorry ', string)
    string = re.sub("so+ ", ' so ', string)
    string = re.sub("lie+ ", ' lie ', string)
    string = re.sub("okay+ ", ' okay ', string)
    string = re.sub(' lol[a-z]+ ', 'laugh out loud', string)
    string = re.sub(' wow+ ', ' wow ', string)
    string = re.sub('wha+ ', ' what ', string)
    string = re.sub(' ok[a-z]+ ', ' ok ', string)
    string = re.sub(' u+ ', ' you ', string)
    string = re.sub(' wellso+n ', ' well soon ', string)
    review_text = re.sub(' byy+ ', ' bye ', string)
    review_text = re.sub("(im\s)+", " i am ", review_text)
    review_text = re.sub("(\wl\ss\w)+", ' also ', review_text)
    review_text = re.sub("(IM\s)+", " i am ", review_text)
    review_text = re.sub("(\sbro$)+", " brother ", review_text)
    review_text = re.sub("\stv", " Television ", review_text)
    # review_text = review_text.replace('’', '\'').replace('"', ' ').replace("`", "'")

    review_text = abbreviation_to_text(review_text)

    string = review_text.replace('whats ', 'what is ').replace("i'm ", 'i am ')
    string = string.replace("it's ", 'it is ')
    string = string.replace('Iam ', 'I am ').replace(' iam ', ' i am ').replace(' dnt ', ' do not ')
    string = string.replace('I ve ', 'I have ').replace('I m ', ' I\'am ').replace('i m ', 'i\'m ')
    string = string.replace('Iam ', 'I am ').replace('iam ', 'i am ')
    string = string.replace('dont ', 'do not ').replace('google.co.in ', ' google ').replace(' hve ', ' have ')
    string = string.replace(' F ', ' Fuck ').replace('Ain\'t ', ' are not ').replace(' lv ', ' love ')
    string = string.replace(' ok~~ay~~ ', ' okay ').replace(' Its ', ' It is').replace(' its ', ' it is ')
    string = string.replace('  Nd  ', ' and ').replace(' nd ', ' and ').replace('i ll ', 'i will ')

    # string = ' ' + string
    # string = abbreviation_to_text(string)
    string = ' ' + string
    for item in LOGOGRAM.keys():
        string = string.replace(' ' + item + ' ', ' ' + LOGOGRAM[item] + ' ')

    list_str = ekphrasis_config(string)
    for index in range(len(list_str)):
        if list_str[index] in EMOTICONS_TOKEN.keys():
            list_str[index] = EMOTICONS_TOKEN[list_str[index]][1:len(EMOTICONS_TOKEN[list_str[index]]) - 1]

    for index in range(len(list_str)):
        if list_str[index] in LOGOGRAM.keys():
            list_str[index] = LOGOGRAM[list_str[index]]

    for index in range(len(list_str)):
        if list_str[index] in LOGOGRAM.keys():
            list_str[index] = LOGOGRAM[list_str[index]]

    string = ' '.join(list_str)
    # review_text = re.sub("(@[\w]*\ )+", " @USER ", string)

    # duplicateSpacePattern = re.compile(r'\ +')
    # review_text = re.sub(duplicateSpacePattern, ' ', review_text).strip()
    # print(review_text)

    # review_text = ekphrasis_config(review_text)
    # print(review_text)
    review_text = re.sub("[^a-zA-Z0-9\@\&\:]", " ", string)

    # review_text = review_text.lower()

    words = stanford_tokenizer(review_text)

    return (words)
for i in sentences:
    res = review_to_wordlist(i)
    print(res)