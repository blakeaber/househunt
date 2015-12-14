
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        PACKAGES                         #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
import pandas as pd
import time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        FUNCTIONS                        #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# RESEARCH: get the lat/long coordinates for each address
# --------------------------------------------------------
def get_coords(addr_df): 

    # Replace the API key below with a valid API key.
    import pandas as pd
    import googlemaps
    gmaps = googlemaps.Client(key='AIzaSyAfqpH0H6e8yKxcRXCLolyY1mxpKGhCmZc')

    # get lat/long results
    save_index = addr_df.index
    addrlist = addr_df.values
    lats, longs = [], []
    for addr in addrlist:
        addr = addr.encode('ascii')

        # Geocoding and address
        geocode_result = gmaps.geocode(addr)
        lat = geocode_result[0]['geometry']['location']['lat']
        long = geocode_result[0]['geometry']['location']['lng']

        # return lists
        lats.append(lat)
        longs.append(long)

    final = pd.concat([addr_df.reset_index(drop=True), pd.Series(lats), pd.Series(longs)], axis=1 ,ignore_index=True)
    final.index = save_index
    return final

# RESEARCH: vectorize columns and hierarchical cluster
# --------------------------------------------------------
def similar(match_list):
    
    # import package
    from difflib import SequenceMatcher
    counter = 0

    # list.pop(0) after every loop to avoid double work
    with open('similarities.txt', 'a') as outfile:
        outfile.write("%s\t%s\t%s\n" % ('col1', 'col2', 'prob'))
        while counter < len(match_list):
            col1 = match_list.pop(0)
            for col2 in match_list:
                prob = SequenceMatcher(None, col1, col2).ratio()
                if prob > 0.75:
                    outfile.write("%s\t%s\t%s\n" % (col1.encode('ascii'), col2.encode('ascii'), prob))
            counter += 1

# RESEARCH: parsing column values for better data
# --------------------------------------------------------
def parse_flat(data, name):

    # copy the original dataframe
    data = data.copy()

    # split column values and stack into one column
    many = data[name].apply(pd.Series).stack()

    # determine substrings with meaning    
    substrings = [i.encode('ascii') for i in many.unique()]

    # check for frequent substrings with meaning
    for ss in substrings:
        filter = many.apply(lambda x: True if x == ss else False)

        # add a new column based on substrings
        # NOTE: this is SUPER SLOW.... needs to be fixed
        new_col = pd.DataFrame(many[filter].notnull().values.astype(int), index=[i[0] for i in many[filter].index])
        new_col.columns = [name + '_' + ss.replace(' ','-')]

        # merge new column onto existing dataframe
        data = data.merge(new_col, how='left', left_index=True, right_index=True)

    # drop the original column
    data = data.drop(name, 1)
    return data

# RESEARCH: parsing column values for better data
# --------------------------------------------------------
def parse_nested(data, name):

    # copy the original dataframe
    data = data.copy()

    # split column values and stack into one column
    one = data[name].apply(pd.Series).stack()

    # split the strings into lists
    many = one.apply(lambda x: x.split(': ')).reset_index()

    # custom rules for common items
    if name == 'facts':
        many[0] = many[0].apply(lambda x: ['built',x[0][-5:]] if x[0][:-4]=='built in ' else x)
        many[0] = many[0].apply(lambda x: ['time-since-listing',x[0][:-15]] if x[0][-14:]=='days on zillow' else x)

    if name == 'features':
        many[0] = many[0].apply(lambda x: ['finished basement'] if x[0][:19]=='finished basement, ' else x)
        many[0] = many[0].apply(lambda x: ['partial basement'] if x[0][:18]=='partial basement, ' else x)
        many[0] = many[0].apply(lambda x: ['basement'] if x[0][-14:]==' sqft basement' else x)

    if name == 'other':
        many[0] = many[0].apply(lambda x: ['built',x[0][-5:]] if x[0][:-4]=='built in ' else x)

    # determine substrings with meaning    
    substrings = [i.encode('ascii') for i in many[0].apply(lambda x: x[0]).unique()]

    # check for frequent substrings with meaning
    for ss in substrings:
        filter = many[0].apply(lambda x: True if x[0] == ss else False)

        # add a new column based on substrings
        ss_filter = many[filter]
        ss_filter.set_index('level_0', inplace=True)
        ss_filter = ss_filter.drop(['level_1'], 1)[0]


        new_col = pd.DataFrame(ss_filter.apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else 1))
        new_col.columns = [name + '_' + ss.replace(' ','-')]

        # merge new column onto existing dataframe
        data = data.merge(new_col, how='left', left_index=True, right_index=True)

    # drop the original column
    data = data.drop(name, 1)
    return data

# - - - - - - - - - - - - - - - - - - - - - - - - - -#
#                        CODE                        #
# - - - - - - - - - - - - - - - - - - - - - - - - - -#

# start time
# --------------------------------------------------------
start_time = time.time()

# import New Canaan JSON data
# --------------------------------------------------------
data = pd.read_json('data.json', convert_axes=False).transpose().reset_index()
time_spent = time.time() - start_time
print('...imported data in', time_spent)
start_time = time.time()

# get lists of strings blown out to columns
# --------------------------------------------------------
data = parse_flat(data, 'appliances included')
time_spent = time.time() - start_time
print('...parsed "appliances included" header in', time_spent)
start_time = time.time()

data = parse_flat(data, 'room types')
time_spent = time.time() - start_time
print('...parsed "room types" header in', time_spent)
start_time = time.time()

# get lists of tuple-strings blown out to columns
# --------------------------------------------------------
data = parse_nested(data, 'facts') # THIS PRODUCES A TON OF COLUMNS
time_spent = time.time() - start_time
print('...parsed "facts" header in', time_spent)
start_time = time.time()

data = parse_nested(data, 'other')
time_spent = time.time() - start_time
print('...parsed "other" header in', time_spent)
start_time = time.time()

data = parse_nested(data, 'features')
time_spent = time.time() - start_time
print('...parsed "features" header in', time_spent)
start_time = time.time()

data = parse_nested(data, 'construction')
time_spent = time.time() - start_time
print('...parsed "construction" header in', time_spent)
start_time = time.time()

# HEADLINE: get lists of csv-strings blown out to columns
# --------------------------------------------------------
counter = 0
data['house_acres'] = data['addr_headline'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else None)
for col in ['beds','baths','sqft']:
    data['house_' + col] = data['addr_headline'].apply(lambda x: x[counter] if isinstance(x, list) and len(x) > 1 else None)
    counter += 1
data = data.drop('addr_headline', 1)
time_spent = time.time() - start_time
print('...parsed "addr_headline" header in', time_spent)
start_time = time.time()

# MORTGAGE / RENT: get lists of csv-strings blown out to columns
# --------------------------------------------------------
for head in ['mortgage','rent']:
    counter = 0
    for col in ['low','mean','high']:
        data[head + '_' + col] = data[head].apply(lambda x: x[counter] if isinstance(x, list) and len(x) > 1 else None)
        counter += 1
    data = data.drop(head, 1)
time_spent = time.time() - start_time
print('...parsed "mortgage" and "rent" headers in', time_spent)
start_time = time.time()

# CONSOLIDATE DATA: combine columns to get cleaner data
# --------------------------------------------------------
data['features_fancy-tub']         = data['features_hot-tub/spa'] + \
                                     data['features_jetted-tub']
data['features_fancy-security']    = data['features_controlled-access'] + \
                                     data['features_disability-access'] + \
                                     data['features_doorman'] + \
                                     data['features_gated-entry'] + \
                                     data['features_intercom'] + \
                                     data['features_security-system']
data['features_fancy-gym']         = data['features_basketball-court'] + \
                                     data['features_fitness-center'] + \
                                     data['features_sports-court']
data['features_fancy-outdoor']     = data['features_garden'] + \
                                     data['features_greenhouse'] + \
                                     data['features_lawn'] + \
                                     data['features_fenced-yard']

# need to combine house_baths & facts_bath
# need to combine facts_parking & features_parking
# need to combine facts_built & other_built
# need to combine house_sqft & other_floor-size

time_spent = time.time() - start_time
print('...consolidated like columns in', time_spent)
start_time = time.time()

# STRING REMOVAL: unlock numerical data from strings
# --------------------------------------------------------
data = data.fillna(0)

def column_cleaner(data, col, func):
    
    # copy the original dataframe
    data = data.copy()

    # apply the function to the column
    data[col] = data[col].apply(func)
    return data

def dummify(data, col, func):
    
    # copy the original dataframe
    data = data.copy()

    # apply the function to the column
    x = column_cleaner(data, col, string_space_split)[col]
    gd = pd.get_dummies(x, prefix=col, dummy_na=True)    

    # merge new column onto existing dataframe
    data = data.merge(gd, how='left', left_index=True, right_index=True)

    # drop the original column
    data = data.drop(col, 1)
    return data

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    import numpy as np

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

# create lambda functions for column parsing
string = lambda x: str(x)
string_comma_int = lambda x: int(str(x).replace(',',''))
string_int = lambda x: int(str(x))
string_space_split = lambda x: str(x).replace(' ','').split(',')[0]

# clean columns with lambda functions
data = column_cleaner(data, 'facts_all-time-views', string_comma_int)
data = column_cleaner(data, 'facts_views-since-listing', string_comma_int)
data = column_cleaner(data, 'other_last-remodel-year', string_int)
data = column_cleaner(data, 'facts_time-since-listing', string_int)
data = column_cleaner(data, 'facts_deposit-&-fees', string_int)
data = column_cleaner(data, 'construction_stories', string_int)
data = column_cleaner(data, 'other_built', string_int)
data = column_cleaner(data, 'facts_built', string_int)

# create dummy variables from categorical data
data = dummify(data, 'facts_laundry', string_space_split)
data = dummify(data, 'features_flooring', string_space_split)
data = dummify(data, 'other_heating', string_space_split)
data = dummify(data, 'features_view', string_space_split)
data = dummify(data, 'facts_pets', string_space_split)
data = dummify(data, 'facts_cooling', string_space_split)
data = dummify(data, 'facts_heating', string_space_split)
data = dummify(data, 'construction_structure-type', string_space_split)
data = dummify(data, 'construction_roof-type', string_space_split)
data = dummify(data, 'facts_lease', string)
data = dummify(data, 'construction_exterior-material', string_space_split)
data = dummify(data, 'other_laundry', string_space_split)
data = dummify(data, 'construction_room-count', string_int)

# parse critical house data
data['house_baths'] = data['house_baths'].apply(lambda x: float(str(x).replace(' baths','').replace(' bath','').replace('-','0')))
data['house_baths'].apply(lambda x: float(x))
data['house_beds'] = data['house_beds'].apply(lambda x: int(str(x).replace(' beds','').replace(' bed','').replace('studio','-').replace('-','0')))
data['house_sqft'] = data['house_sqft'].apply(lambda x: int(str(x).replace(' sqft','').replace(',','').replace('-','0')))
data['facts_lot'] = data['facts_lot'].apply(lambda x: eval(str(x).replace(' sqft','*43560').replace(',','').replace(' acres','').replace(' acre','')))

time_spent = time.time() - start_time
print('...added dummy columns in', time_spent)
print('...ALL DONE!!!')


# IGNORE: drop sparse (with little value) features
# --------------------------------------------------------
data = data.drop('open house', 1) # Spare feature that probably isn't useful
data = data.drop('index', 1) # Spare feature that probably isn't useful
data = data.drop('additional features', 1) # Spare feature that probably isn't useful
data = data.drop('sale_headers', 1) # Spare feature that probably isn't useful
data = data.drop('sale_values', 1) # Spare feature that probably isn't useful
data = data.drop('tax_headers', 1) # Spare feature that probably isn't useful
data = data.drop('tax_values', 1) # Spare feature that probably isn't useful
data = data.drop('other_parcel-#', 1) # Spare feature that probably isn't useful
data = data.drop('facts_posted', 1) # Spare feature that probably isn't useful

data = data.drop('rent_low', 1) # duplicate value - needs to be consolidated
data = data.drop('rent_mean', 1) # duplicate value - needs to be consolidated
data = data.drop('rent_high', 1) # duplicate value - needs to be consolidated
data = data.drop('mortgage_low', 1) # duplicate value - needs to be consolidated
data = data.drop('mortgage_high', 1) # duplicate value - needs to be consolidated
data = data.drop('facts_baths', 1) # duplicate value - needs to be consolidated
data = data.drop('house_acres', 1) # duplicate value - needs to be consolidated
data = data.drop('facts_parking', 1) # duplicate value - needs to be consolidated
data = data.drop('features_parking', 1) # duplicate value - needs to be consolidated
data = data.drop('other_floor-size', 1) # duplicate value - needs to be consolidated
data = data.drop('facts_hoa-fee', 1) # duplicate value - needs to be consolidated

# check that all columns are numerical
# num_cols = list(data.describe().transpose()['count'].index)
# all_cols = list(data.columns)
# str_cols = list(set(all_cols)-set(num_cols))
# data[str_cols].head()

# - - - - - - - - - - - - - - - - - - - - - - - - - -#
#                   LOCATION DATA                    #
# - - - - - - - - - - - - - - - - - - - - - - - - - -#
geo = pd.read_csv('all_addr_geocoded.txt', sep='\t', index_col=0)
data = data.merge(geo, how='left', left_on='addr', right_on='addr')
data['dist_from_train1'] = haversine_np(data['long'], data['lat'], -73.49627405, 41.14588886)
data['dist_from_train2'] = haversine_np(data['long'], data['lat'], -73.49778414, 41.11589603)
data['dist_from_retail'] = haversine_np(data['long'], data['lat'], -73.49334508, 41.14683816)
data['dist_from_hs'] = haversine_np(data['long'], data['lat'], -73.48935664, 41.12871426)
data = data.drop('addr', 1) # Spare feature that probably isn't useful

# - - - - - - - - - - - - - - - - - - - - - - - - - -#
#                  DATA EXPLORATION                  #
# - - - - - - - - - - - - - - - - - - - - - - - - - -#
appliance_col = [i for i in data.columns if i[:10]=='appliances']
construction_col = [i for i in data.columns if i[:12]=='construction']
facts_col = [i for i in data.columns if i[:5]=='facts']
features_col = [i for i in data.columns if i[:8]=='features']
other_col = [i for i in data.columns if i[:5]=='other']
room_col = [i for i in data.columns if i[:10]=='room types']




pick_col = appliance_col[0]
def binary_plot(pick_col):
    x_0 = data[data[pick_col]==0][pick_col]
    x_1 = data[data[pick_col]==1][pick_col]
    y = data['mortgage_mean'].str.replace(r'\D+', '').replace(r'', '0').astype('int')

    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot([y[x_0.index],y[x_1.index]], showfliers=False)
    ax.set_title(pick_col)
    plt.show()

for i in range(0,len(construction_col)):
    binary_plot(construction_col[i])




# Create a set of independent variables / outcome for feature selection
import numpy as np
from sklearn.cross_validation import train_test_split

mdl = data.copy()
mdl = mdl[(mdl.mortgage_mean != 'unavailable') & (mdl.mortgage_mean != 0)]
mdl['mortgage_mean'] = mdl['mortgage_mean'].str.replace(r'\D+', '').astype('int')
mdl = mdl[(mdl.mortgage_mean >= 500000) & (mdl.mortgage_mean <= 3500000)]

y = np.log(mdl['mortgage_mean'])
X = mdl.drop('mortgage_mean', 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=61)





import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.
lin_model = linear_model.LinearRegression()
rfecv = RFECV(estimator=lin_model, step=1, cv=KFold(X_train.shape[0],10), scoring='mean_squared_error')
rfecv.fit(X_train, y_train)

# print feature ranking
rfecv.ranking_

# Predict data of estimated models
y_preds = rfecv.predict(X_test)
plt.plot(y_test, y_preds, '.g')
plt.legend(loc='lower right')
plt.show()






# Fit line using all data
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Robustly fit linear model with RANSAC algorithm
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(X_train, y_train)
inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
y_preds = model.predict(X_test)
y_preds_ransac = model_ransac.predict(X_test)

# print score results
model_ransac.score(X_test,y_test)
model.score(X_test,y_test)

# Compare R
plt.plot(X_train[inlier_mask], y_train[inlier_mask], '.g', label='Inliers')
plt.plot(X_train[outlier_mask], y_train[outlier_mask], '.r', label='Outliers')
plt.show()

plt.plot(y_test, y_preds, '.b', label='Linear regressor')
plt.plot(y_test, y_preds_ransac, '.r', label='RANSAC regressor')
plt.legend(loc='lower right')
plt.show()












# Feature Importance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=100,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(15), importances[indices][:15],
       color="r", yerr=std[indices][:15], align="center")
plt.xticks(range(15), indices[:15], rotation=-45)
plt.xlim([-1, 15])
plt.show()






# - - - - - - - - - - - - - - - - - - - - - - - - - -#
#                    RESEARCH                        #
# - - - - - - - - - - - - - - - - - - - - - - - - - -#

# LET'S TRY BUILDING A MODEL !!!
mdl = data.copy()
mdl = mdl[(mdl.mortgage_mean != 'unavailable') & (mdl.mortgage_mean != 0)]
mdl['mortgage_mean'] = mdl['mortgage_mean'].str.replace(r'\D+', '').astype('int')
mdl = mdl[(mdl.mortgage_mean >= 500000) & (mdl.mortgage_mean <= 3500000)]

mdl=mdl[mdl.describe().transpose()['mean'][mdl.describe().transpose()['mean']>1].index]

mdl[['addr','lat','long','mortgage_mean']].to_csv('fusion_newcanaan.csv', header=['addr','lat','long','mortgage_mean'], index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.cross_validation import train_test_split

y = mdl['mortgage_mean']
stats.probplot(y, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()

X = mdl.drop('mortgage_mean', 1)
X = (X - X.mean()) / (X.std() + 0.001)
X['intercept'] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=61)

est = sm.OLS(y_train, X_train).fit()
est.summary()

coef = est.params
p_vals = est.pvalues

preds = est.predict(X_test)
plt.scatter(preds, (preds-y_test))
plt.xlabel("Predicted")
plt.ylabel("Observed")

fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(est, 'facts_built', ax=ax)

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(est, fig=fig)






# FIND SIMILAR TEXT STRINGS WITH THIS FUNCTION
# --------------------------------------------------------
# similar(list(data.columns))

# PULL THE DATA IN CHUNKS FROM GOOGLE
# NOTE: DAYS ARE CALENDAR, NOT 24 HOUR... SO PULL DATA AT 11:50 AND 12:10 TO GAME QUOTAS
# --------------------------------------------------------
# results = get_coords(data['addr'][:2490])   ...DONE!!!
# results = results[0]
# results.columns = ['addr','lat','long']
# results.to_csv('2490addr_geocoded.txt', sep='\t')
# results = get_coords(data['addr'][2490:])
# results = results[0]
# results.columns = ['addr','lat','long']
# results.to_csv('4723addr_geocoded.txt', sep='\t')

# WORK AROUND INDICES
# --------------------------------------------------------
# addrs = results['addr'][results['addr'].notnull()].reset_index()
# lats = results['lat'][results['lat'].notnull()].reset_index(drop=True)
# longs = results['long'][results['long'].notnull()].reset_index(drop=True)
# temp = pd.concat([addrs,lats,longs], axis=1, ignore_index=True).set_index(0)
# temp.columns = ['addr','lat','long']

# GET BOUNDING BOXES FOR NEW CANAAN REAL ESTATE
# --------------------------------------------------------
# latlongs = pd.read_csv('all_addr_geocoded.txt', sep='\t', index_col=0)
# dsc = latlongs.describe(percentiles=[.05,.25,.5,.75,.95])

# lats = [dsc['lat']['5%'], dsc['lat']['25%'], dsc['lat']['50%'], dsc['lat']['75%'], dsc['lat']['95%']]
# longs = [dsc['long']['5%'], dsc['long']['25%'], dsc['long']['50%'], dsc['long']['75%'], dsc['long']['95%']]

# import itertools as it
# combos = list(it.product(lats, longs))

# GOOGLE SEARCH API FOR LAT/LONG LOCATIONS
# --------------------------------------------------------
# https://maps.googleapis.com/maps/api/place/nearbysearch/output?parameters

# from googleplaces import GooglePlaces, types, lang
# google_places = GooglePlaces('AIzaSyAfqpH0H6e8yKxcRXCLolyY1mxpKGhCmZc')

# query_result = google_places.place_search(
# query_result = google_places.nearby_search(
#         location='41.1463,-73.49563', radius=5000, types=[types.TYPE_FOOD])






