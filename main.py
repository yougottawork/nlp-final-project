import string

# with the file we created, score each comment with the guidelines below
# After when we're done with scoring, we can sort the comments in decreasing order

#Remove all basic words using stoplist

stopwords = ['a','the','The','an','and','or','but','about','above','after','along','amid','among', 'as','at','by','for','from','in','into','like','minus','near','of','off','on', 'onto','out','over','past','per','plus','since','till','to','under','until','up', 'via','vs','with','that','can','cannot','could','may','might','must', 'need','ought','shall','should','will','would','have','had','has','having','be','is','am','are','was','were','being','been','get','gets','got','gotten','getting','seem','seeming','seems','seemed','enough', 'both', 'all', 'your' 'those', 'this', 'these','their', 'the', 'that', 'some', 'our', 'my','its', 'his' 'her', 'each', 'any', 'another','an', 'a', 'just', 'such', 'right', 'not','only', 'sheer', 'namely', 'as', 'same', 'different', 'such','when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which','whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace','anything', 'anytime' 'anywhere', 'everyplace', 'everything' 'everywhere', 'whatever', 'whereever', 'whichever', 'he','him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs','you','your','yours','me','my','mine','I','i','we','us','much','and/or']

useful_wordfreq = {}
funny_wordfreq = {}
cool_wordfreq = {}

#never, too, very, --> extreme word list
    # score each of those words
        #never, -3 points, 
        #very, +2 points
    # for each review, calculate the extremity score
        # Ratings, the number of votes, etc
count=0

with open("review_yelp_sample.csv", encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(1, len(lines)-1): # from 2nd line
        if (lines[i][0].isdigit()) :
            useful_tags = int(lines[i][2])
            funny_tags = int(lines[i][4])
            cool_tags = int(lines[i][6])
            total_tags = useful_tags + funny_tags + cool_tags
            if (total_tags > 0) :
                wordlist2 = []
                wordlist = lines[i][8:].split()
                for word in wordlist:
                    for character in word:
                        if character in string.punctuation:
                            word = word.replace(character,"")
                            word = word.lower() # to lowercase the words
                    wordlist2.append(word)
                    # wordlist2 will have all the words of the comment
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

sorted_useful = dict( sorted(useful_wordfreq.items(),key=lambda item: item[1],reverse=True))
sorted_funny = dict( sorted(funny_wordfreq.items(),key=lambda item: item[1],reverse=True))
sorted_cool = dict( sorted(cool_wordfreq.items(),key=lambda item: item[1],reverse=True))

print(sorted_useful)
print("\n")
print(sorted_funny)
print("\n")
print(sorted_cool)
print("\n")
                
# reference
    # https://rstudio-pubs-static.s3.amazonaws.com/117743_53a6a1da3c724e7b8d608c3df7e40fb6.html#/3
    # 

#Scoring = extremity + ratings of the review + # of votes + 
        
#Create a list of common words using the leftover words
#POS tag the common words in the list

#proper nouns
    #Restaurants, manager names, the name of the menu/cuisine, 

#word net

# uspe first program to get ne, w csv, which becomes data ffor next  programfilor\