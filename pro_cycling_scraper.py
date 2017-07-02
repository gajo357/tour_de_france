'''
Created on Jun 29, 2017

@author: Gajo
'''
from bs4 import BeautifulSoup
import requests
import json
from utils import convert_to_float, convert_to_int
import pandas as pd
import re 
from IPython.core import payload
from astropy.io.ascii.cparser import pprint

pro_site = 'http://www.procyclingstats.com/'
pattern = re.compile('[\W_]+', re.UNICODE)

rider_labels = ['id', 'rider_name', 'height', 'weight', 'one_day', 'gc', 'time_trial', 'sprint', 'pcs_rank', 'uci_rank']
stages_labels = ['id', 'stage_name', 'length', 'type'] ##, 'avg_speed'
short_labels = ['rider_id', 'stage_id', 'result', 'pcs', 'uci']
all_labels = ['season','rider_name','weight','height','pcs_rank','uci_rank','one_day','gc','time_trial','sprint','stage_result','stage_pcs','stage_uci','stage_name','stage_lenth''stage_type'] #,'stage_avg_speed',
    
def read_rider_info(rider_soup, rider):
    weight = None
    height = None
    pcs = None
    uci = None
    for b in rider_soup.find_all('b'):
        if b.contents[0] == 'Weight:':
            weight = convert_to_float(b.next_sibling.replace('kg', ''))
            continue
        if b.contents[0] == 'Height:':
            height = convert_to_float(b.next_sibling.replace('m', ''))
            r = b.parent.contents[6]
            pcs = convert_to_int(r.contents[1])
            uci = convert_to_int(r.contents[3])
            continue
        if weight is not None and height is not None:
            break
    
    rankings = rider_soup.find_all('div', class_='pbsRow')
    one_day = rankings[0].parent.contents[1].contents[0]
    gc = rankings[1].parent.contents[1].contents[0]
    time_trial = rankings[2].parent.contents[1].contents[0]
    sprint = rankings[3].parent.contents[1].contents[0]
    
    rider['weight'] = weight        
    rider['height'] = height
    rider['pcs_rank'] = pcs
    rider['uci_rank'] = uci        
    rider['one_day'] = one_day
    rider['gc'] = gc
    rider['time_trial'] = time_trial
    rider['sprint'] = sprint

def read_stage_name(stage_soup):
    ## remove all non alphanumeric, unicode safe
    stage_info = stage_soup.find('div', class_='entryHeader').find('h2')                
    stage_name = pattern.sub('', stage_info.contents[2].contents[0]).encode('utf-8').upper()
    return stage_name

def read_stage_info(stage_soup, stages, season, stage_length):
    try:
        stage_name = read_stage_name(stage_soup)
        stage_name += '_' + season
        exists = True
        if stage_name not in stages:
            exists = False
            info = stage_soup.find('div', class_='subDiv info show') 
            #stage_avg_speed = None
            stage_type = None
            try:       
                #stage_avg_speed = convert_to_float(info.find(string='Avg. speed winner:').parent.next_sibling.replace('km/h', ''))
                pt_tag = info.find("span", text=lambda text: not text)        
                stage_type = pt_tag['class'][0]
            except:
                print(stage_name)
                    
            stages[stage_name] = {}
            stages[stage_name]['stage_name'] = stage_name
            stages[stage_name]['id'] = len(stages)
            stages[stage_name]['length'] = stage_length
            #stages[stage_name]['avg_speed'] = stage_avg_speed
            stages[stage_name]['type'] = stage_type
            
        return (exists, stages[stage_name])
    except:
        return (False, None)
    
def append_rider_data(all_results, rider):
    all_results.append(rider['rider_name'])
    all_results.append(rider['weight'])      
    all_results.append(rider['height'])
    all_results.append(rider['pcs_rank'])
    all_results.append(rider['uci_rank'])        
    all_results.append(rider['one_day'])
    all_results.append(rider['gc'])
    all_results.append(rider['time_trial'])
    all_results.append(rider['sprint'])
    
def append_stage_data(all_results, stage):
    all_results.append(stage['stage_name'])
    all_results.append(stage['length'])
    #all_results.append(stage['avg_speed'])
    all_results.append(stage['type'])
    
def scrape_all_data():
    session = requests.session()

    # get all info from the race page
    req = session.get(pro_site + 'race.php?id=171088&c=3&code=race-startlist')
    soup = BeautifulSoup(req.content, 'lxml')
    
    seasons = ['2017', '2016']

    riders = []
    stages = []
    stages_dict = {}
    all_results = []
    short_results = []
    
    # loop over all riders on the page
    rider_links = soup.find_all('a', { "class" : "rider" })
    for i in range(0, len(rider_links)):
        link = rider_links[i]
        
        rider_name = ' '.join((link.contents[0].contents[0].strip(), link.contents[1].strip())).encode('utf-8').upper()
        print('Processing rider {0} : {1} of {2}'.format(rider_name, i+1, len(rider_links)))
        
        rider = {}
        rider['rider_name'] = rider_name
        riders.append(rider)
        rider['id'] = len(riders)
        
        # go to rider's page
        riders_page = pro_site + link.get('href')
        req = session.get(riders_page)
        rider_soup = BeautifulSoup(req.content, 'lxml')
        
        read_rider_info(rider_soup, rider)
                
        # check all needed seasons
        for season in seasons:
            ## get all results
            req = session.get(riders_page + '&season=' + season)
            races_soup = BeautifulSoup(req.content, 'lxml')
            
            result_rows = races_soup.find_all('div', class_= 'row light')            
            for result_row in result_rows:
                ## take data from results row
                stage_result = convert_to_int(result_row.contents[1].contents[0])
                distance = convert_to_float(result_row.contents[5].contents[0])
                if stage_result is None or distance is None:
                    continue
                stage_pcs = convert_to_int(result_row.contents[6].contents[0], default=0)
                stage_uci = convert_to_int(result_row.contents[7].contents[0], default=0)
                                              
                ## read the information about the stage
                req = session.get(pro_site + result_row.a.get('href'))
                stage_soup = BeautifulSoup(req.content, 'lxml')
                exists, stage = read_stage_info(stage_soup, stages_dict, season, distance)
                if stage is None:
                    continue
                if not exists:
                    stages.append(stage)
                
                ## put result data into results
                result = []
                result.append(season)
                append_rider_data(result, rider)
                result.append(stage_result)
                result.append(stage_pcs)
                result.append(stage_uci)
                append_stage_data(result, stage)
                all_results.append(result)
                
                short_result = {}
                short_result['rider_id'] = rider['id']
                short_result['stage_id'] = stage['id']
                short_result['result'] = stage_result
                short_result['pcs'] = stage_pcs
                short_result['uci'] = stage_uci
                short_results.append(short_result)     
                
    riders_df = pd.DataFrame(riders)
    stages_df = pd.DataFrame(stages)
    results_df = pd.DataFrame.from_dict(short_results)
    #print(riders_df.head())
    #print(stages_df.head())
    #print(results_df.head())
    riders_df.to_csv('riders1.csv', index=False)
    stages_df.to_csv('stages1.csv', index=False)
    results_df.to_csv('short_results1.csv', index=False)
    
    riders_df = riders_df[rider_labels]
    stages_df = stages_df[stages_labels]
    results_df = results_df[short_labels]    
    #print(riders_df.head())
    #print(stages_df.head())
    #print(results_df.head())
    
    riders_df.to_csv('riders.csv', index=False)
    stages_df.to_csv('stages.csv', index=False)
    results_df.to_csv('short_results.csv', index=False)
    
    all_results_df = pd.DataFrame.from_records(all_results, columns=all_labels)
    all_results_df.to_csv('full_result.csv', index=False)
    #with open('full_result.csv', 'w') as f:
    #    f.write("{}\n".format('season,rider_name,weight,height,pcs_rank,uci_rank,one_day,gc,time_trial,sprint,stage_result,stage_pcs,stage_uci,stage_name,stage_lenth,stage_avg_speed,stage_type'))
    #    for item in all_results:
    #        f.write("{}\n".format(','.join(item)))
    pass

def scrape_tour_de_france_stages():
    stages = []
    
    # get all info from the race page
    session = requests.session()
    req = session.get(pro_site + 'race.php?id=171088&c=4')
    soup = BeautifulSoup(req.content, 'lxml')
    
    table = soup.find('table', class_='basic')
    for tr in table.find_all('tr')[1:]:
        tds = tr.find_all('td')
        td = tds[1].a
        link = td.get('href')

        ## read the information about the stage
        req = session.get(pro_site + link)
        stage_soup = BeautifulSoup(req.content, 'lxml')
        stage_name = read_stage_name(stage_soup)
        stages.append(stage_name)
        
    with open('tdf_stages.txt', 'w') as f:
        for item in stages:
            f.write("{}\n".format(item))
    return stages

def scrape_manager():
    #site = 'https://www.holdet.dk/da/tour-de-france-2017/'
    
    ht = 'https://www.holdet.dk/handlers/tradedata.ashx?game=tour-de-france-2017&page=page_Id&r=r_Id'
    payloads = [1498992552420, 1498992692724, 1498992724104, 1498992735593, 1498992747598, 1498992760018, 1498992781743, 1498992795986]
    
    riders = []
    riders_dict = {}
    # get all info from the race page
    session = requests.session()
    for i in range(0, 8):
        site = ht.replace('&page=page_Id&r=r_Id', '&page={}&r={}'.format(i, payloads[i]))
        req = session.get(site)
        data = json.loads(req.content)
        for item in data['Dataset']['Items']:
            name = item['Values'][2].encode('utf-8').upper()
            value = item['Values'][16]
            if name not in riders_dict:
                riders.append({'rider_name' : name, 'cost' : value})
                riders_dict[name] = value
    
    riders_df = pd.DataFrame(riders)
    riders_df.to_csv('riders_cost.csv', index=False)
    pass

if __name__ == '__main__':
    scrape_manager()
    pass