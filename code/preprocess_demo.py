#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/4 10:18
import re

def emoji_to_text(review_text):

    # review_text = str(review_text)
    review_text = review_text.replace("👊", " Oncoming Fist ")
    review_text = review_text.replace("😂", " Face With Tears of Joy ")
    review_text = review_text.replace(" 🎶 ", " Musical Notes ")
    review_text = review_text.replace("🎶 ", " Musical Notes ")
    review_text = review_text.replace("♥️",  " Heart Suit ")
    review_text = review_text.replace("♥", " Heart Suit ")
    review_text = review_text.replace("👏", " Clapping Hands ")
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
    review_text = review_text.replace("🙄", " Face With Rolling Eyes ").replace("☺", " Smiling Face ")
    review_text = review_text.replace("💚", " Green Heart ")
    review_text = review_text.replace("🤠", " Cowboy Hat Face ")
    review_text = review_text.replace("👍", " Thumbs Up ")
    review_text = review_text.replace("❤", " Red Heart ")
    review_text = review_text.replace("❤️", " Red Heart ")
    # review_text = review_text.replace("*", " Asterisk ")
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
    review_text = review_text.replace("😫", " Tired Face ")
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
    review_text = review_text.replace("😆", " Grinning Squinting Face ").replace("😥", " Sad but Relieved Face ").replace("😧", " Anguished Face ")
    review_text = review_text.replace("😎", " Smiling Face With Sunglasses ")
    review_text = review_text.replace("🤷‍♀", " Woman Shrugging ")
    review_text = review_text.replace("😚", " Kissing Face With Closed Eyes ")
    review_text = review_text.replace("🙌🏾", " Raising Hands: Medium-Dark Skin Tone ")
    review_text = review_text.replace("😳", " Flushed Face ").replace("🏾‍♀️", " Woman Bouncing Ball ")
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
    review_text = review_text.replace("🆗", " ok ")
    review_text = review_text.replace("🤭", " Face With Hand Over Mouth ")
    review_text = review_text.replace("😘", " Face Blowing a Kiss ")
    review_text = review_text.replace("🎂", " Birthday Cake ")
    review_text = review_text.replace("🙌", " Raising Hands ")
    review_text = review_text.replace("😪", " Sleepy Face ")
    review_text = review_text.replace("🐇", " Rabbit ")
    review_text = review_text.replace("🕳️", " Hole ")
    review_text = review_text.replace("%", "percent")
    review_text = review_text.replace("😡", " Pouting Face ")
    review_text = review_text.replace("🙏🏻", " Folded Hands: Light Skin Tone ")
    review_text = review_text.replace("💙", " Blue Heart ").replace("🎉", " Party Popper ")
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
    review_text = review_text.replace("🤷🏻‍♀️", " Woman Shrugging: Light Skin Tone ")
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
    review_text = review_text.replace("✝", " Latin Cross ")
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
    review_text = review_text.replace("🤦🏾‍♀️", " Woman Facepalming: Medium-Dark Skin Tone ").replace("🤦🏼‍♂️", " Man Facepalming: Medium-Light Skin Tone ")
    review_text = review_text.replace("👋", " Waving Hand ").replace(" 🏻 ", " Light Skin Tone ")
    review_text = review_text.replace("🐾", " Paw Prints ")
    review_text = review_text.replace("💀", " Skull")
    review_text = review_text.replace("😋", " Face Savoring Food ")
    review_text = review_text.replace("🤦🏽‍♂", " Man Facepalming: Medium Skin Tone ")
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
    review_text = review_text.replace("💀", " Skull ")
    review_text = review_text.replace("😩", " Weary Face ")
    review_text = review_text.replace("🤷🏼‍♂", " Man Shrugging: Medium-Light Skin Tone ")
    review_text = review_text.replace("🤷🏼‍♂️", " Man Shrugging: Medium-Light Skin Tone ")
    review_text = review_text.replace("🤦‍♀", " Woman Facepalming ")
    review_text = review_text.replace("🤦‍♀️", " Woman Facepalming ")
    review_text = review_text.replace("💪🏻", " Flexed Biceps: Light Skin Tone ")
    review_text = review_text.replace("💩", " Pile of Poo")
    review_text = review_text.replace("💉", " Syringe ")
    review_text = review_text.replace("😱"," Face Screaming in Fear ")
    review_text = review_text.replace("😠 "," Angry Face ")
    review_text = review_text.replace("🌍", " Globe Showing Europe-Africa ")
    review_text = review_text.replace("👮‍♀️", " Woman Police Officer ")
    review_text = review_text.replace("👮‍♀", " Woman Police Officer ")
    review_text = review_text.replace("💰", " Money Bag ")
    review_text = review_text.replace("💓", " Beating Heart ")
    review_text = review_text.replace("💖", " Sparkling Heart ").replace("♡", " White Heart Suit ")
    review_text = review_text.replace("😏", " Smirking Face ").replace("😜", " Winking Face With Tongue ").replace("🙁", " Slightly Frowning Face ")
    review_text = review_text.replace("🤷🏾‍♀", " Woman Shrugging: Medium-Dark Skin Tone ")
    review_text = review_text.replace("💁🏽‍♀️", " Woman Tipping Hand: Medium Skin Tone ")
    review_text = review_text.replace("💁🏽‍♀", " Woman Tipping Hand: Medium Skin Tone ")
    review_text = review_text.replace("🌚", " New Moon Face ").replace("😝", " Squinting Face With Tongue ").replace("😮", " Face With Open Mouth ")
    review_text = review_text.replace("🤯", " Exploding Head ")
    review_text = review_text.replace("👇", " Backhand Index Pointing Down ")
    review_text = review_text.replace("🤧", " Sneezing Face ")
    review_text = review_text.replace("🐒", " Monkey ")
    review_text = review_text.replace("🤫", " Shushing Face ")
    review_text = review_text.replace("🤷🏼‍♀", " Woman Shrugging: Medium-Light Skin Tone ")
    review_text = review_text .replace("🤪", " Zany Face ")
    review_text = review_text.replace("🤗", " Hugging Face ")
    review_text = review_text.replace("💗", " Growing Heart ")
    review_text = review_text .replace("🙃", " Upside-Down Face ")
    review_text = review_text .replace("🐷", " pig ")
    review_text = review_text.replace("🙇🏾‍♀", " Woman Bowing: Medium-Dark Skin Tone ")
    review_text = review_text.replace("🙇🏾‍♀️", " Woman Bowing: Medium-Dark Skin Tone ")
    review_text = review_text.replace("🤡", " Clown Face ")
    review_text = review_text.replace("🎥", " Movie Camera ")
    review_text = review_text.replace("▶", " play ")
    review_text = review_text.replace("◀ ", " Reverse ")
    review_text = review_text.replace("🏼 ", " Medium-Light Skin Tone ")
    review_text = review_text.replace("🤑", " Money-Mouth Face " )
    review_text = review_text.replace("🌞", " sun with face ")
    review_text = review_text.replace("🚂", " Locomotive ")
    review_text = review_text.replace(" 🐕", " dog ")
    review_text = review_text.replace("🤦🏾‍♂️", " Man Facepalming: Medium-Dark Skin Tone ")
    review_text = review_text.replace("🐖", " pig ")
    review_text = review_text.replace("😞", " Disappointed Face ")
    review_text = review_text.replace("💦", " Sweat Droplets ")
    review_text = review_text.replace("😿", " Crying Cat Face ").replace("🤮", " Face Vomiting ").replace("😠", " Angry Face ")
    review_text = review_text.replace("🤦🏽‍♀️", " Woman Facepalming: Medium Skin Tone ").replace("🏻‍♂️", " Woman Bouncing Ball: Light Skin Tone ")
    review_text = review_text.replace("👑", " Crown ").replace("😦", " Frowning Face With Open Mouth ")
    review_text = review_text.replace("💁🏽‍♂️", " Man Tipping Hand: Medium Skin Tone ").replace(" 🦇!", " bat ")
    review_text = review_text.replace("🇬🇧", " Flag: United Kingdom ").replace("🏴‍☠", " Pirate Flag ")
    review_text = review_text.replace("🇺🇸", " Flag: United States ").replace("🏼‍♀️", " Man Bouncing Ball: Medium-Dark Skin Tone ")
    review_text = review_text.replace("🤤", " Drooling Face ").replace("😖", " Confounded Face ").replace("🍽", " Fork and Knife With Plate ")
    review_text = review_text.replace("🇨🇦", " canada ")
    review_text = review_text.replace("🤷🏽‍♂️", " Woman Shrugging: Medium Skin Tone ")
    review_text = review_text.replace(" 📍", " Round Pushpin ")
    review_text = review_text.replace(" 🙌🏻", " Raising Hands: Light Skin Tone ").replace("👽", " Alien ").replace("🌃 ", " Night With Stars ")
    review_text = review_text.replace(" 🏁", " Chequered Flag ").replace("✅", " White Heavy Check Mark ")
    review_text = review_text.replace("🌻", " Sunflower ").replace("😹", " Cat Face With Tears of Joy ")
    review_text = review_text.replace("🚫"," Prohibited").replace("🖤", " black heart ").replace("🍕", " Pizza ")
    review_text = review_text.replace("✊🏾", " Raised Fist ").replace("💴", " Yen Banknote ").replace("💁🏽 "," Person Tipping Hand ").replace("😃", " Grinning Face With Big Eyes ")
    review_text = review_text.replace(" 🤦‍♂️" ," Man Facepalming ").replace("🏼", " Medium-Light Skin Tone ")
    review_text = review_text.replace("😬", " Grimacing Face ").replace("👆", " Backhand Index Pointing Up ")
    review_text = review_text.replace("❗", " Exclamation Mark ").replace(" 🏻‍♀️", " Woman Bouncing Ball ").replace(" 🏽‍♂️", " man Bouncing Ball ")
    review_text = review_text.replace("😐", " Neutral Face ").replace("💵", " Dollar Banknote ")
    # review_text = " ".join(review_text)
    return review_text


def abbreviation_to_text(review_text):

    # review_text = re.sub("\s(em)\s", " them ", review_text)
    # review_text = re.sub("(im\s)+"," i am ", review_text)
    # review_text = re.sub("(\wl\ss\w)+", ' also ', review_text)
    # review_text = re.sub("(IM\s)+", " i am ", review_text)
    # review_text = re.sub("(\sbro$)+", " brother ", review_text)
    # review_text = re.sub("\stv", " Television ", review_text)


    review_text = review_text.replace('’', '\'').replace('"', ' ').replace("`", "'")
    review_text = review_text.replace("you’re", ' you are ').replace("You’re", ' You are ').replace("You're", ' You are ').replace("you're", " you are ")
    review_text = review_text.replace("wouldn't", " would not ").replace("aren’t", " are not ")
    # review_text = review_text.replace("You’re", 'You are')
    # review_text = review_text.replace("You're", 'You are')
    review_text = review_text.replace("should'veTaken", " should've Taken ").replace("we'd", " we should ")
    review_text = review_text.replace("should've", " should have ")
    review_text = review_text .replace("won't", " won not ")
    review_text = review_text.replace("it’s", " it is ").replace("It’s", " It is ").replace("It's"," It is ").replace("it's", " it is ")
    # review_text = review_text.replace("It’s", "It is")
    # review_text = review_text.replace("It's","It is")
    review_text = review_text.replace("doesn’t", " dose not ").replace("Doesn’t", " Does not ")
    review_text = review_text.replace("What’s", " What is ").replace("what's", " what is ").replace("what’s", " what is ").replace("What's", " what is ")
    # review_text = review_text.replace(" what's", "what is")
    review_text = review_text.replace("you've", " you have ").replace(" w/u ", " you ")
    review_text = review_text.replace("he’s", " he is ").replace("He’s", " He is ").replace("He's", " He is ").replace("he's", " he is ").replace("he’d", " he should ")
    # review_text = review_text.replace("He’s", "He is")
    review_text = review_text.replace("There's", " There is ").replace("there's", " there is ").replace("there’s", " there is ")
    # review_text = review_text.replace(" there's ", "there is")
    review_text = review_text.replace("women's", " woman is ").replace("mom’s", " mom is ").replace("woman’s", " woman is ")
    review_text = review_text .replace("they’re", " they are ").replace("ain’t", " am not ").replace("God’s", " god is ").replace("God's", " god is ").replace("Ain’t", " am not ")
    review_text = review_text.replace("men's", " men is ").replace("weren't", " were not ")
    review_text = review_text.replace("That's", " That is ").replace("that’s", " that is ").replace("that's", " that is ").replace("That's", " that is ")
    # review_text = review_text.replace("that’s", "that is")
    review_text = review_text.replace(" She's ", " she is ").replace("she’s", " she is ").replace("she's", " she is ")
    # review_text = review_text.replace("she’s", "she is")
    review_text = review_text.replace("didn’t", " did not ").replace("don't", " do not ").replace("don’t", " do not ").replace("Don't", " Do not ").replace("doesn't", " does not ").replace("Don’t", " do not ").replace("didn't", " did not ").replace("Doesn't", " does not ")
    # review_text = review_text.replace("don't", "do not")
    # review_text = review_text.replace("don’t", "do not")
    # review_text = review_text.replace("Don't", "Do not")
    review_text = review_text.replace("I'll"," I will ").replace("I’ll", " I will ")
    # review_text = review_text.replace("I’ll", "I will")
    review_text = review_text.replace("I'd", " I would ").replace("i’d", " i would ").replace("I’d", " I would ").replace("they'd", " they would ")
    review_text = review_text.replace(" I’ve ", " I have ")
    review_text = review_text.replace("I’m", " I am ").replace("i’m", " i am ").replace("I'm", " I am ").replace("I've", " i have ").replace("i'm", " i am ")
    # review_text = review_text.replace("i’m", "i am")
    review_text = review_text.replace("Let's", " Let is ").replace("Let’s", " let is ").replace("let’s", " let is ")
    review_text = review_text.replace("Isn't", " is not ")
    # review_text = review_text.replace(" time’s ", " time is ")
    review_text = review_text.replace("won’t", " won not ")
    review_text = review_text.replace("can’t", " can not ").replace("can't", " can not ").replace("Can't", " can not ")
    review_text = review_text.replace("hadn't", " had not ")
    review_text = review_text.replace("wouldn’t", " would not ")
    review_text = review_text.replace("Shouldn't", " Should not ").replace("shouldn't", " should not ").replace("shouldn’t", " should not ")
    # review_text = review_text.replace("shouldn't", "should not")
    # review_text = review_text.replace("shouldn’t", "should not")
    review_text = review_text.replace("aren't", " are not ").replace("It'd", " it would ")
    review_text = review_text.replace("wasn't", " was not ")
    review_text = review_text.replace("She’s", " She is ").replace("she’s", " she is ").replace("she's", " she is ").replace("prick's", " prick is ")
    # review_text = review_text.replace("she’s", "she is")
    review_text = review_text.replace("She’ll", " She will ").replace("she’ll", " she will ").replace("She's", " she is ").replace("hasn’t", " has not ")
    # review_text = review_text.replace("she’ll", "she will")
    review_text = review_text.replace("Trump’s", " Trump is ").replace("it’ll", " it will ")
    review_text = review_text.replace("Couldn't", " Could not ").replace("couldn't", " could not ")
    review_text = review_text.replace("isn’t", " is not ").replace("Isn't", " Is not ").replace("isn't", " is not ")
    review_text = review_text.replace("Can't", " Can not ").replace("CAN'T", " can not ").replace("can't", " can not ")
    review_text = review_text.replace("weren’t", " were not ").replace("They're", " they are ").replace("they're", " they are ")
    review_text = review_text.replace(" they'll ", " they will ").replace(" TV ", " Television ")
    review_text = review_text.replace("f**king", " fuck").replace("f**k", " fuck ").replace("f**ked", " fucked ")
    review_text = review_text.replace("#Trump2020"," trump 2020 ").replace("Ted's", " Ted is ").replace("Thomas's", " Thomas is ").replace("Trump's", " Trump is ")
    # review_text = review_text.replace("#Trump2020", " trump 2020 ")
    review_text = review_text.replace("#TrudeauMustGo", " trudeau must go ")
    review_text = review_text.replace("#DeepStateCorruption", " deep state corruption ")
    review_text = review_text.replace("#PutUpOrShutUp", " put up or shut up ")
    review_text = review_text.replace("#CallTheVoteAlready", " Call The Vote Already ")
    review_text = review_text.replace("Flynn’s", " Flynn is ").replace("#DangerousDemocrats", " Dangerous Democrats ")
    review_text = review_text.replace("#KavanaughConfirmation", " Kavanaugh Confirmation ")
    review_text = review_text.replace("#WakeUpAmerica", " Wake Up America ").replace("#CloserNation", " Closer Nation ")
    review_text = review_text.replace("#LeviStrauss", " Levi Strauss ")
    review_text = review_text.replace("sh*t", " shit ").replace("who's", " who is ")
    review_text = review_text.replace("#CrookedHillary", " Crooked Hillary ").replace("SJWs", " Social Justice Warriors ")
    review_text = review_text.replace("#ChristineBlaseyFord", " Christine Blasey Ford ")
    review_text = review_text.replace("#ConfirmKavanaugh", " Confirm Kavanaugh ")
    review_text = review_text.replace("fascist's", " fascist is ")
    review_text = review_text.replace("follower's", " follower is")
    review_text = review_text.replace("ball's", " ball is ").replace("Woodword's", " Woodword is ")
    review_text = review_text.replace("#machinelearning", " machine learning")
    review_text = review_text.replace("#ThursdayThoughts", " Thursday Thoughts ")
    review_text = review_text.replace("Non-voters", " Nonvoters ")
    review_text = review_text.replace("community's", " community is ")
    review_text = review_text.replace("Here's", " here is ").replace("here's", " here is ")
    review_text = review_text.replace("#DrainTheDeepState", " Drain The Deep State ")
    review_text = review_text.replace("#DrainTheSwamp", " Drain The Swamp ")
    review_text = review_text.replace("We've", " We have ").replace("We're", " we are ").replace("we're", " we are ")
    review_text = review_text.replace("#brexitshambles", " brexit shambles ")
    review_text = review_text.replace("#BrexitMeansBrexit", " Brexit Means Brexit ")
    review_text = review_text.replace("you'll", " you will ")
    review_text = review_text.replace("#oshaeterry", " oshae terry ")
    review_text = review_text.replace("#ThugLife", " Thug Life ")
    review_text = review_text.replace("#MAGA2020", " MAGA 2020 ")
    review_text = review_text.replace("#MAGARallyRules", " MAGA Rally Rules ")
    review_text = review_text.replace("#TexasPolice", " Texas Police ")
    # review_text = review_text.replace("IM", " i am ").replace("im", " i am ")
    review_text = review_text.replace("didnt", " did not ").replace("Gemini's", " Gemini is ").replace("Y'all", " you and all ")
    review_text = review_text.replace("#proudtobeBritish", " proud to be British ")
    review_text = review_text.replace("OMG", " Oh My God! ")
    review_text = review_text.replace("#FrankOz", " Frank Oz ")
    review_text = review_text.replace("KKK", " Ku Klux Klan ").replace("LMAOOOO", "lmao").replace("LMAOOO", "lmao").replace("Lmaoooo", "lmao")
    review_text = review_text.replace("lmao", " Laughing My Ass Off ")
    review_text = review_text.replace("Lmao", " Laughing My Ass Off ")
    review_text = review_text.replace("LMAO", " Laughing My Ass Off ")
    review_text = review_text.replace("Dems", " Development Engineering Management System ").replace("dems", " Development Engineering Management System ")
    review_text = review_text.replace("uhm", " universal host machine ")
    review_text = review_text.replace("#ScotusKavenaugh", " Scotus Kavenaugh ")
    review_text = review_text.replace("#REDWAVERISING2018", " RED WAVE RISING 2018 ").replace("#NightMayor 's", " Night Mayor is ").replace("Nenshi’s", " Nenshi is ")
    # review_text = review_text.replace("#REDWAVERISING2018", " RED WAVE RISING 2018 ").replace("#NightMayor 's", " Night Mayor is ")
    review_text = review_text.replace("#fordnation", " ford nation ")
    review_text = review_text.replace("YESSSSSHHHHH", " yes ").replace("Lol", "lol")
    review_text = review_text.replace("lol"," Laugh Out Loud ")
    review_text = review_text.replace("CEO", " Chief Executive Officer ")
    review_text = review_text.replace("COO", " Chief Operating Officer ")
    review_text = review_text.replace("CBS", " Columbia Broadcasting System " )
    review_text = review_text.replace("needa", " need ").replace("Devil's", " Devil is ")
    # review_text = review_text.replace("needa", " need ")
    review_text = review_text.replace(" Ur ", " ultra rare ").replace(" u ", " you ")
    review_text = review_text.replace("scotus", " Supreme Court of the United States ").replace("Fuck u", " fuck you ")
    review_text = review_text.replace("fuckin", " fuck ").replace("y'all's", " you and all is ")
    review_text = review_text.replace("peopl ehave", " people have ").replace(" CA ", " Canada ").replace("GOP", " Grand Old Party ").replace("CNN"," Cable News Network ")
    review_text = review_text.replace("Budden's", " Budden is ").replace("M*neta's", "  Moneta is ")
    review_text = review_text.replace("haven’t", " have not ").replace("She'll", " she will ")
    review_text = review_text.replace(" Tbh ", " to be honest ").replace(" tbh ", " to be honest ")
    review_text = review_text.replace("CUCK", " a political neologism and term of abuse:white nationalists to insult Republican politicians who are too mainstream")
    review_text = review_text.replace(" EU "," European Union ")
    review_text = review_text.replace("LOL", " laughing out loud ").replace("wanna", " want to ")
    review_text = review_text.replace("stick w it", " stick with it ").replace("SJW", " Social justice warrior ")
    review_text = review_text.replace("GDP", " Gross Domestic Product ").replace("IQ", " intelligence quotient ")
    review_text = review_text.replace(" fe ", " feel ")
    review_text = review_text.replace("#AI ", " artificial intelligence ").replace("#StrataData", " Strata Data ")
    review_text = review_text.replace(" em ", " them ")
    review_text = review_text.replace(" ea silly ", " easilly ")
    review_text = review_text.replace("S he ", " she ")
    review_text = review_text.replace("ESPN", " Entertainment Sports Programming Network ")
    review_text = review_text.replace("DiDNT" , " doesn't exist ")
    review_text = review_text.replace("y'all", " you all ")
    review_text = review_text.replace(" omg ", "oh my god ")
    review_text = review_text.replace("POS"," point-of-sale ")
    review_text = review_text.replace("Gonna", " going to ").replace(" U ", " you ")
    review_text = review_text.replace("vict i am", " victiam ").replace(" b ", " be ")
    review_text = review_text.replace("Heart-E yes", " Heart-Eyes ").replace("#VoteBlue2018", " Vote Blue 2018 ")
    review_text = review_text.replace("soci also curity", " social so curity").replace("cla i am", " claiam")
    review_text = review_text.replace("#YESonKavanaugh", " YES on Kavanaugh ").replace("NDP", " New Democratic Party ")
    review_text = review_text.replace("Smh", " Shaking My Head ").replace("WAPO", " White American Political Organization ",)
    review_text = review_text.replace("SS", " Schutzstaffel ").replace("Mm", " millimeter ").replace("KP", " Kilometer Post ")
    review_text = review_text.replace("steepppp", " step ").replace("yyyy", " you ").replace("#NotPlastic", " Not Plastic ")
    review_text = review_text.replace("#StopKavanaugh", " Stop Kavanaugh ").replace("#StopTrump", " Stop Trump ").replace("#SaveSCOTUS", " Save SCOTUS ")
    review_text = review_text.replace(" fuck g ", " fucking ").replace(" yep ", " yes ").replace(" bc "," British Columbia ")
    review_text = review_text.replace("MF"," millifarad ").replace("DNC", " Democratic National Committee ").replace("#BetoNotForTexas", " Beto Not For Texas ")
    review_text = review_text.replace("Shes", " she is ").replace("shes", " she is ").replace("RIP"," rest in peace ")
    review_text = review_text.replace("h i am", " him ").replace("#UFC228", " UFC 228 ")
    review_text = review_text.replace("UFC", " Ultimate Fighting Championship ").replace(" DC ", " District of Columbia ",)
    review_text = review_text.replace("🇳🇬", " Nigeria ").replace("re-form", " reform").replace("CIA", " Central Intelligence Agency ").replace('NPR', "National Public Radio",)
    review_text = review_text.replace("IRS", " Internal Revenue Service ").replace("bullsh#t", " bullshit ")
    review_text = review_text.replace("Didn't", " did not ").replace("idk"," I don not know")
    review_text = review_text.replace("CUUUUUKS", " cuks ").replace("f*ck", " fuck ").replace("T i am", "tim").replace("CFO"," Chief Financial Officer ")
    review_text = review_text.replace("#Googlearecorrupt", " Google are corrupt ").replace("#BestuseQwant", " Best use Queen want ")
    review_text = review_text.replace("NFL"," National Football League ").replace("CBC"," Canadian Broadcasting Corporation ")
    review_text = review_text.replace(" DT", " Data Technology").replace("b*******", " bitch ").replace("#LeftWingLiberalDisease", " Left Wing Liberal Disease ")
    review_text = review_text.replace("#WalkAway", " Walk Away ").replace("F*******", " fuck ")
    review_text = review_text.replace(" s***", " shit ").replace("he'd", " he should ").replace("#MeToo", " me too ").replace("haven't", " have not ")
    review_text = review_text.replace("#PatriotsUnited", " Patriots United ").replace("#WeAreQ", " We Are Q ").replace("THERE'S", " there is ")
    review_text = review_text.replace("Wil lie", " Willie ").replace(" FSU "," Former Soviet Union ").replace("They've", " they have ")
    review_text = review_text.replace("#WalkAwayFromDemocrats2018", " Walk Away From Democrats 2018 ").replace("SHOULDN'T ", " should not ").replace("E very "," every " )
    review_text = review_text.replace("#DregOfSociety", " Dreg Of Society ").replace("#VoteThemOut2018", " Vote Them Out 2018 ")
    review_text = review_text.replace("#TrumpsArmy", " Trumps Army ").replace("#ShameOnYou", " Shame On You ").replace("smh ", " Shaking My Head ")
    review_text = review_text.replace("We'll", " we will ").replace("we've", " we have ").replace(" FBI ", " Federal Bureau of Investigation ")
    review_text = review_text.replace("#FortTrump", " Fort Trump ").replace("#BoomingEconomy", " Booming Economy ").replace("A also e", " all she ")
    review_text = review_text.replace(" BS"," British Standard").replace("You've" ," you have ").replace("#doSomthing", " do Somthing ")
    review_text = review_text.replace("#CountryOverParty", " Country Over Party ").replace(" Omg ", " oh my god ").replace("shyt", " shit ")
    review_text = review_text.replace(" IM ", " i am ").replace(" im ", "i am ").replace("MGK "," Medieval Greek ").replace("Myyy", " my ")
    review_text = review_text.replace("#DeepStatePanic", " Deep State Panic ").replace(" OCD ", " obsessive-compulsive disorder ")
    review_text = review_text.replace(" St "," standard time ")
    review_text = review_text.replace("mi*rosoft", " microsoft ").replace("#GreatestThingsAboutThe90s", " Greatest Things About The 90s ")
    review_text = review_text.replace("bitchhhhhh", " bitch ").replace("bitchhh", " bitch ").replace(" Bf ", " boyfriend ")
    review_text = review_text.replace("#NeverForget", " Never Forget ").replace(" Bro ", " brother ").replace("let's", " let is ")
    review_text = review_text.replace("Fuuckkkk", " fuck ").replace("youuuuu", " you ").replace(" em", " them ")
    # review_text = " ".join(review_text)
    return review_text

