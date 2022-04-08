import string
import nltk
from nltk.tag import pos_tag
stopwords = ['a','the','The','an','and','or','but','about','above','after','along','amid','among', 'as','at','by','for','from','in','into','like','minus','near','of','off','on', 'onto','out','over','past','per','plus','since','till','to','under','until','up', 'via','vs','with','that','can','cannot','could','may','might','must', 'need','ought','shall','should','will','would','have','had','has','having','be','is','am','are','was','were','being','been','get','gets','got','gotten','getting','seem','seeming','seems','seemed','enough', 'both', 'all', 'your' 'those', 'this', 'these','their', 'the', 'that', 'some', 'our', 'my','its', 'his' 'her', 'each', 'any', 'another','an', 'a', 'just', 'such', 'right', 'not','only', 'sheer', 'namely', 'as', 'same', 'different', 'such','when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which','whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace','anything', 'anytime' 'anywhere', 'everyplace', 'everything' 'everywhere', 'whatever', 'whereever', 'whichever', 'he','him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs','you','your','yours','me','my','mine','I','i','we','us','much','and/or']

useful_wordfreq = {}
funny_wordfreq = {}
cool_wordfreq = {}

extremewords = ['very', 'too', 'extremely', 'really','totally', 'furious',
    'awful','terrible','horrible','filthy','wonderful','fantastic','excellent',
        'fascinating','gorgeous','terrifying','disgusting','best','worst','rude',
        'fuck','fucking','fucked','freaking','so','dick','ass','absolutely','absolute','shit',
            'trash','omg','incredible','unbelievable','incredibly','unbelievably',
                'wow','phenomenal','awesome','definitely','never','ever','greatest',
                    'most','least','always','horrendous','must']

#never, too, very, --> extreme word list
    # score each of those words
        #never, -3 points, 
        #very, +2 points
    # for each review, calculate the extremity score
        # Ratings, the number of votes, etc
count=0

# input = open("review_yelp.csv", encoding='utf-8')
# output = open("review_output.csv", 'w', newline='', encoding='utf-8')

# output.write(input.readline())

# for line in input:
#     if not line.lstrip().startswith("#"):
#         output.write(line)

# input.close()
# output.close()

with open("review_yelp.csv") as f, open("reviews_weighted.txt", 'w') as g:
    lines = f.readlines()
    g.write("Stars\tUseful\tFunny\tCool\tReview\tExtreme_words_ratio\tnum_of_proper_nouns\tnum_of_capped_words\n")
    for i in range(1, len(lines)-1): # from 2nd line
        weight=0
        ratingweight=0 #if rate is 1 or 5, we weight it 3 . If rate is 2 or 4, we weight it 2. If rate is 3, we weight it 1.
        capword=0
        numword=0
        propernouns=0 #number of proper nouns in the comment will be counted
        extword=0
        if (lines[i][0].isdigit()) :
            useful_tags = int(lines[i][2])
            funny_tags = int(lines[i][4])
            cool_tags = int(lines[i][6])
            total_tags = useful_tags + funny_tags + cool_tags
            #if (total_tags > 0) :
            wordlist2 = []
            wordlist = lines[i][8:].split()
            for word in wordlist:
                for character in word:
                    if character in string.punctuation:
                        word = word.replace(character,"")
                wordlist2.append(word)
            numword = len(wordlist2)
            for word in wordlist2:
                if word.lower() in extremewords:
                    extword += 1
            for word in wordlist2:
                str1 = " "
                for ele in wordlist2:
                    str1+= " " + ele
                tagged = nltk.tag.pos_tag((str1).split())
                edited_sentence = [word for word,tag in tagged if tag == 'NNP']
                propernouns = len(edited_sentence)
            for word in wordlist2:
                if (word.isupper() and len(word) != 1 and word != 'OK' and word not in edited_sentence):
                    capword += 1
            most_tags = max(useful_tags , funny_tags , cool_tags)
            if (most_tags == useful_tags) :
                for word in wordlist2:
                    if word not in stopwords:
                        if word in useful_wordfreq: # word frequency is stored here
                            useful_wordfreq[word] += 1
                        else:
                            useful_wordfreq[word] = 1
            if (most_tags == funny_tags) :
                # do the same but for funny_tags category
                for word in wordlist2:
                    if word not in stopwords:
                        if word in funny_wordfreq: # word frequency is stored here
                            funny_wordfreq[word] += 1
                        else:
                            funny_wordfreq[word] = 1
            if (most_tags == cool_tags) :
                # do the same but for cool_tags category
                for word in wordlist2:
                    if word not in stopwords:
                        if word in cool_wordfreq: # word frequency is stored here
                            cool_wordfreq[word] += 1
                        else:
                            cool_wordfreq[word] = 1   
            extreme_word_ratio = (extword/numword)
            extreme_word_ratio = round(extreme_word_ratio, 3)
            g.write(lines[i][0] + '\t' + lines[i][2] + '\t' + lines[i][4] + '\t' + lines[i][6] + '\t' + lines[i][8:].strip('\n') + '\t' + str(extreme_word_ratio) + '\t' + str(propernouns) + '\t' + str(capword) + '\n') 
