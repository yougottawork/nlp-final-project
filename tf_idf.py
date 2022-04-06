import pandas as pd 
line_counter = -1
with open("review_yelp.txt", errors = "ignore") as f:
    # print(f)
    for line in f:
        if line_counter <= 10: 
            print(line)
            line_counter += 1

# from encodings.aliases import aliases
# alias_values = set(aliases.values())

# for encoding in set(aliases.values()):
#     try:
#         df=pd.read_excel("review_yelp.xlsx", encoding=encoding)
#         print('successful', encoding)
#     except:
#         print('failed', encoding)
#         pass