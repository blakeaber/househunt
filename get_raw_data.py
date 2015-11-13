

# import python packages
# --------------------------------------------------------
from bs4 import BeautifulSoup
from selenium import webdriver
import simplejson as json
import requests
import time

# function: get_soup()
# --------------------------------------------------------
def get_soup(url):
    """
    This function is used to mask http requests with user agents and pull
    the html data for parsing
    """
    browser = webdriver.Chrome('/Users/blakeaber/Desktop/NewCanaanResearch/chromedriver')
    browser.get(url)
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    soup = BeautifulSoup(browser.page_source,'lxml')
    browser.quit()
    return soup

# loop through matches for each name
# --------------------------------------------------------
completed = 0
id_timer = 0
counter = 57325478
while counter <= 57328880:

    # keep a dictionary of all results
    # --------------------------------------------------------
    query_results = {}

    # track the average time to pull each search
    # --------------------------------------------------------
    id_start = time.time()

    # pull landing page for each name search
    # --------------------------------------------------------
    soup = get_soup('http://www.zillow.com/homedetails/%s_zpid/' % counter)

    # start results dict
    # --------------------------------------------------------
    interim_results = {}
    
    # save the property summary
    # --------------------------------------------------------
    if soup.find_all(attrs={ 'class' : 'hdp-header-description'}):
        overview = [s for s in soup.find_all(attrs={ 'class' : 'hdp-header-description'})]
        interim_results['addr'] = ' '.join([s.h1.get_text().strip() for s in overview[0].find_all(attrs={ 'class' : 'addr'})])
        interim_results['addr_headline'] = [s.get_text().strip() for s in overview[0].find_all(attrs={ 'class' : 'addr_bbs'})]
    
    # save the property details
    # --------------------------------------------------------
    if soup.find_all(attrs={ 'class' : 'fact-group-container'}):
        facts = [s for s in soup.find_all(attrs={ 'class' : 'fact-group-container'})]
        sections = [(fact.h3.get_text(), [l.get_text() for l in fact.ul.find_all('li')]) for fact in facts]
        for k,v in sections:
            interim_results[k] = v
        
    # save the property values
    # --------------------------------------------------------
    if soup.find_all(attrs={ 'data-module' : 'zestimate'}):
        zestimate = soup.find('div', { 'class' : 'zest-container'})
        buy_low, buy_cost, buy_high, buy_change, rent_low, rent_cost, rent_high, rent_change = 0, 0, 0, 0, 0, 0, 0, 0
        if len(zestimate.find_all(attrs={ 'class' : 'zest-value'})) >= 2:
            buy_cost, rent_cost = [z.get_text() for z in zestimate.find_all(attrs={ 'class' : 'zest-value'})][:2]
        if len(zestimate.find_all(attrs={ 'class' : 'zest-change'})) >= 2:
            buy_change, rent_change = [z.get_text() for z in zestimate.find_all(attrs={ 'class' : 'zest-change'})]
        if len(zestimate.find_all(attrs={ 'class' : 'zest-range-bar-low'})) >= 2:
            buy_low, rent_low = [z.get_text() for z in zestimate.find_all(attrs={ 'class' : 'zest-range-bar-low'})]
        if len(zestimate.find_all(attrs={ 'class' : 'zest-range-bar-high'})) >= 2:
            buy_high, rent_high = [z.get_text() for z in zestimate.find_all(attrs={ 'class' : 'zest-range-bar-high'})]
        interim_results['mortgage'] = [buy_low, buy_cost, buy_high, buy_change]
        interim_results['rent'] = [rent_low, rent_cost, rent_high, rent_change]

    # save the tax info
    # --------------------------------------------------------
    if soup.find('div', { 'id' : 'hdp-tax-history'}):
        taxes = soup.find('div', { 'id' : 'hdp-tax-history'}).table
        if taxes:
            tax_headers = [i.get_text() for i in taxes.thead.find_all('th')]
            tax_values = [[j.get_text() for j in i.find_all('td')] for i in taxes.tbody.find_all('tr')]
            interim_results['tax_headers'] = tax_headers
            interim_results['tax_values'] = tax_values

    # save the sales info
    # --------------------------------------------------------
    if soup.find('div', { 'id' : 'hdp-price-history'}):
        sales = soup.find('div', { 'id' : 'hdp-price-history'}).table
        if sales:
            sale_headers = [i.get_text() for i in sales.thead.find_all('th')]
            sale_values = [[j.get_text() for j in i.find_all('td')][:-1] for i in sales.tbody.find_all('tr')]
            interim_results['sale_headers'] = sale_headers[:-1]
            interim_results['sale_values'] = sale_values

    # drop soup object
    # ----------------------------------------------------
    soup.decompose()

    # save dictionary to master results
    # ----------------------------------------------------
    query_results[counter] = interim_results

    # print completed IDs
    # ----------------------------------------------------    
    if interim_results:
        print(str(counter) + " : " + query_results[counter]['addr'])
    else:
        print(str(counter))

    # increment counters
    # ----------------------------------------------------
    counter += 1 # 4.7K addresses
    completed += 1
    
    # save master to file
    # ----------------------------------------------------
    with open('data.json', 'a') as outfile:
        json.dump(query_results, outfile)
        outfile.write('\n')

    # save program times
    # ----------------------------------------------------
    id_timer += time.time() - id_start

# print out program times
# ----------------------------------------------------
print("- - - - - - - - RUN STATS - - - - - - - -")
print("      TOTAL SEARCHES: ", completed)
print("- - - - - - - - - - - - - - - - - - - - -")
print("  SEARCH RUNTIME (s): ", id_timer)
print("- - - - - - - - - - - - - - - - - - - - -")
print("     AVG SEARCH (s): ", id_timer/completed)



