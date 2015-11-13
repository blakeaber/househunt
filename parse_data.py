

# import python packages
# --------------------------------------------------------
import pandas as pd

# RESEARCH: vectorize columns and hierarchical cluster
# --------------------------------------------------------
# def similar(match_list):
#     
#     # import package
#     from difflib import SequenceMatcher
#     counter = 0
# 
#     # list.pop(0) after every loop to avoid double work
#     with open('similarities.txt', 'a') as outfile:
#         outfile.write("%s\t%s\t%s\n" % ('col1', 'col2', 'prob'))
#         while counter < len(match_list):
#             col1 = match_list.pop(0)
#             for col2 in match_list:
#                 prob = SequenceMatcher(None, col1, col2).ratio()
#                 outfile.write("%s\t%s\t%s\n" % (col1.encode('ascii'), col2.encode('ascii'), prob))
#             counter += 1

# RESEARCH: parsing column values for better data
# --------------------------------------------------------
def parse_flat(data, name):

    # split column values and stack into one column
    many = data[name].apply(pd.Series).stack()

    # determine substrings with meaning    
    substrings = many.unique()

    # check for frequent substrings with meaning
    for ss in substrings:
        filter = many.apply(lambda x: True if x == ss else False)

        # add a new column based on substrings
        # NOTE: this is SUPER SLOW.... needs to be fixed
        new_col = pd.DataFrame(many[filter].values, index=[i[0] for i in many[filter].index])
        new_col.columns = [name + '_' + ss.replace(' ','-')]

        # merge new column onto existing dataframe
        data = data.merge(new_col, how='left', left_index=True, right_index=True)

    # drop the original column
    data = data.drop(name, 1)
    return data

# RESEARCH: parsing column values for better data
# --------------------------------------------------------
def parse_nested(data, name):

    # split column values and stack into one column
    one = data[name].apply(pd.Series).stack()

    # split the strings into lists
    many = one.apply(lambda x: x.split(': ')).reset_index()

    # determine substrings with meaning    
    substrings = [i.encode('ascii') for i in many[0].apply(lambda x: x[0]).unique()]

    # check for frequent substrings with meaning
    for ss in substrings:
        filter = many[0].apply(lambda x: True if x[0] == ss else False)

        # add a new column based on substrings
        ss_filter = many[filter]
        ss_filter.set_index('level_0', inplace=True)
        ss_filter = ss_filter.drop(['level_1'], 1)[0]

        new_col = pd.DataFrame(ss_filter.apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else x[0]))
        new_col.columns = [name + '_' + ss.replace(' ','-')]

        # merge new column onto existing dataframe
        data = data.merge(new_col, how='left', left_index=True, right_index=True)

    # drop the original column
    data = data.drop(name, 1)
    return data

# import New Canaan JSON data
# --------------------------------------------------------
data = pd.read_json('data.json', convert_axes=False).transpose().reset_index()

# get lists of strings blown out to columns
# --------------------------------------------------------
data = parse_flat(data, 'facts')

# get lists of tuple-strings blown out to columns
# --------------------------------------------------------
data = parse_nested(data, 'appliances included')
data = parse_nested(data, 'other')
data = parse_nested(data, 'features')
data = parse_nested(data, 'construction')

# get lists of csv-strings blown out to columns
# --------------------------------------------------------
counter = 0
data['house_acres'] = data['addr_headline'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else None)
for col in ['beds','baths','sqft']:
    data['house_' + col] = data['addr_headline'].apply(lambda x: x[counter] if isinstance(x, list) and len(x) > 1 else None)
    counter += 1
data = data.drop('addr_headline', 1)

# very unique columns (or key columns)
data['addr']
data['additional features']


 u'mortgage',
 u'open house',
 u'rent',
 u'room types',
 u'sale_headers',
 u'sale_values',
 u'tax_headers',
 u'tax_values']





data.facts.apply(lambda x: len(x))













