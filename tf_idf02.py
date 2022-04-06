import pandas as pd 
import string 
import nltk
import math

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.stem.porter import * 
ps = PorterStemmer()

from stop_list import closed_class_stop_words as stop_list
review_word_collection = {} # {review_id: [word1, word2, ...]}
review_id = 1
with open("no_header_review_yelp.txt", errors = "ignore") as f:
    # print(f)
    for line in f:
        if review_id <= 5: # Making the first 100 lines the train set 
            # Strip the quotation marks and spaces 
            stars, useful, funny, cool, review = line.strip("").strip().split('\t')
            review_words = review.split()
            for word in review_words: 
                # In here, I strip off all punctuation to isolate the words. 
                # If the word is in the stoplist (keep original capitalization), don't add to the list of words for a review
                # Ignore digits and punctuation characters 
                # If it's a valid word, we will turn it into lowercase and then lemmatize it, then add into review_id word list 
                word = word.strip(string.punctuation)
                # word = ps.stem(word) # Stemming may or may not help. I've chosen not to use it for now
                if word in stop_list: 
                    pass 
                elif word.isdigit() == True or word.isalpha() == False: 
                    pass
                elif (word in string.punctuation) == True: 
                    pass
                else: 
                    if review_id not in review_word_collection: 
                        review_word_collection[review_id] = [] # initialize key to empty list 
                    word = word.lower() # lowercase to standardize everything
                    word = lemmatizer.lemmatize(word)
                    review_word_collection[review_id].append(word) # add in all cases 
            # print(review_word_collection)
                # print(word)
            # print(stars, useful, funny, cool)
            review_id += 1


word_freq = {} # dict of word frequency from review_word_collection key = word, value = count
TFIDF_review = [] # list of dictionaries that record the TFIDF scores for each word per review

for rev_id, rev_word_list in review_word_collection.items():
    for word in rev_word_list: # iterate through each word of the list
        if word not in word_freq: # create key, value if not in word_freq
            word_freq[word] = 1
        else:
            word_freq[word] += 1 # increase count if in word_freq
# print(word_freq)
total_review = 5 # total number of reviews
# print(total_review)


for rev_id, rev_word_list in review_word_collection.items():  # Iterate through key, value in dict
    TFIDF_review.append({})  # Each review will be a dict
    word_base = set()
    for term in rev_word_list:
        if term in word_base:
            TFIDF_review[-1][term] += 1  # increase the count if term is in word base
        else:
            word_base.add(term)
            TFIDF_review[-1][term] = 1  # set the count if term isn't in word base

        term_freq = float(TFIDF_review[-1][term]) / float(len(rev_word_list))  # Term Frequency

        IDF = math.floor(math.log(float(total_review) / float(word_freq[term])) * 1000) / 1000  # IDF: number of documents / number of documents containing the term

        TFIDF_review[-1][term] = math.floor(term_freq * IDF * 1000) / 1000  # TFIDF (term freq) * (Inverse Doc Freq)

print(TFIDF_review)

# from encodings.aliases import aliases
# alias_values = set(aliases.values())

# for encoding in set(aliases.values()):
#     try:
#         df=pd.read_excel("review_yelp.xlsx", encoding=encoding)
#         print('successful', encoding)
#     except:
#         print('failed', encoding)
#         pass