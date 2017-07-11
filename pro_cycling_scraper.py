'''
Created on Jun 29, 2017

@author: Gajo
'''
from bs4 import BeautifulSoup
import requests
import json
from utils import convert_to_float, convert_to_int, process_string
import pandas as pd
import re 
from collections import namedtuple

FullName = namedtuple('FullName', ['name', 'surname'])

pro_site = 'http://www.procyclingstats.com/'
pattern = re.compile('[\W_]+', re.UNICODE)

rider_labels = ['id', 'rider_name', 'height', 'weight', 'one_day', 'gc', 'time_trial', 'sprint', 'pcs_rank', 'uci_rank']
stages_labels = ['id', 'stage_name', 'length', 'type'] ##, 'avg_speed'
short_labels = ['rider_id', 'stage_id', 'result', 'pcs', 'uci']
all_labels = ['season','rider_name','weight','height','pcs_rank','uci_rank','one_day','gc','time_trial','sprint','stage_result','stage_pcs','stage_uci','stage_name','stage_lenth''stage_type'] #,'stage_avg_speed',

def read_time_result(soup):
    try:
        time = soup.find('span', class_='time').text
        hms = time.split(':')
    
        if len(hms) == 1:
            return convert_to_int(hms[0], 0)
        if len(hms) == 2:
            return 60*convert_to_int(hms[0], 0) + convert_to_int(hms[1], 0)
        if len(hms) == 3:
            return 60*60*convert_to_int(hms[0], 0) + 60*convert_to_int(hms[1], 0) + convert_to_int(hms[2], 0)
        
        return 0
    except:
        return 0

def read_team_info(info, teams):
    try:
        team_name = process_string(info[1].contents[0])
        team_link = info[1].get('href')
        if team_name not in teams:
            teams[team_name] = {'name': team_name,
                                'link': team_link,
                                'id': len(teams) + 1}
        return teams[team_name]['id']
    except:
        return 0

def read_rider_info(rider_soup, riders, teams):
    #try:
        position = convert_to_int(rider_soup.find('span').find('span').text)
        if position is None:
            return (None, None, None)
        
        time_lag = read_time_result(rider_soup)        
        
        #read the rider's info
        info = rider_soup.find_all('a')
                           
        team_id = read_team_info(info, teams)
            
        name = process_string(info[0].contents[1])
        surname = process_string(info[0].contents[0].contents[0])
        rider_link = info[0].get('href')
        key = FullName(name = name, surname = surname)
        if key not in riders:
            riders[key] = {'name' : name,
                           'surname': surname,
                           'link': rider_link,
                           'team_id': team_id,
                           'id': len(riders) + 1}
        
        return (time_lag, position, riders[key]['id'])
    #except:
    #    return (None, None, None)

def read_stage_length(stage_soup):
    ## remove all non alphanumeric, unicode safe
    stage_info = stage_soup.find('div', class_='entryHeader').find('h2')
    stage_length = convert_to_float(stage_info.find_all('span')[2].text.replace('(', '').replace(')', '').replace('k', ''))
    return stage_length

def read_stage_info(stage_soup):
    try:
        stage_length = read_stage_length(stage_soup)
        info = stage_soup.find('div', class_='subDiv info show') 
        
        #stage_avg_speed = convert_to_float(info.find(string='Avg. speed winner:').parent.next_sibling.replace('km/h', ''))
        stage_name = process_string(info.find(string='Start/finish:').parent.next_sibling)
        pt_tag = info.find("span", text=lambda text: not text)        
        stage_type = pt_tag['class'][0]      
                
        stage = {'name': stage_name,
                 'length': stage_length,
                 'type': stage_type}
            
        return stage
    except:
        print('error')
        return None

def read_stage_results(soup, race_id, stages, riders, teams, results, stage_day):
    stage = read_stage_info(soup)    
    if stage is None:
        return
    
    stage['id'] = len(stages) + 1
    stage['race_id'] = race_id
    stage['day'] = stage_day
    stages.append(stage)
    
    print(stage)
    
    # read the results table
    winner_time = None
    table = soup.find('div', class_='result')
    for tr in table.find_all('div'):
        (time_lag, position, rider_id) = read_rider_info(tr, riders, teams)
        if position is None:
            continue
        
        if winner_time == None:
            winner_time = time_lag
            time = time_lag
        else:
            time = winner_time + time_lag
            
        results.append({'rider_id': rider_id,
                        'stage_id': stage['id'],
                        'position': position,
                        'time': time})    
    pass

def is_one_day_race(soup):
    info = soup.find('div', class_='entryHeader').find('h2') 
    if info.contents[0].contents[0] == 'One Day Race':
        return True
    return False
    pass

def scrape_all_data():
    """
        Scrape the 'http://www.procyclingstats.com/' website for all info about stages and riders
    """
    race_filter = 'races.php?year=0&circuit=1&ApplyFilter=Filter'
    session = requests.session()    
    
    seasons = ['2017', '2016', '2015', '2014']

    races = []
    riders = {}
    stages = []
    teams = {}
    results = []
    
    for season in seasons:
        print(season)
        
        link = pro_site + race_filter.replace('year=0', 'year={}'.format(season))
        # get all races for the season
        req = session.get(link)
        soup = BeautifulSoup(req.content, 'lxml')
        
        table = soup.find('table', class_ = 'basic')
        for row in table.find_all('tr')[1:]:
            tds = row.find_all('td')
            winner_tag = tds[2].find('a')           
            winner_name = winner_tag.text
            if not winner_name:
                continue
                        
            a = tds[1].find('a')
            race_link = a.get('href')
            race_name = process_string(a.contents[1])
            race_class = tds[3].contents[0]
                        
            print('\nrace: ' + race_name)
            
            race = {'id': len(races) + 1,
                    'name' : race_name,
                    'link' : race_link,
                    'class' : race_class.replace('.UWT', ''),
                    'season' : int(season)}
            races.append(race)
            race_id = race['id']
                        
            req = session.get(pro_site + race_link)
            soup = BeautifulSoup(req.content, 'lxml')
            
            if is_one_day_race(soup):
                race['no_stages'] = 1
                result = soup.find('ul', class_='entryNav').contents[0].find('a').get('href')
                req = session.get(pro_site + result)
                soup = BeautifulSoup(req.content, 'lxml')
                read_stage_results(soup, race_id, stages, riders, teams, results, 1)
            else:
                req = session.get(pro_site + race_link.replace('&c=2', '&c=4'))
                soup = BeautifulSoup(req.content, 'lxml')
                table = soup.find('table', class_='basic')
                table_rows = table.find_all('tr')[1:]
                race['no_stages'] = len(table_rows)
                for i in range(0, len(table_rows)):
                    tr = table_rows[i]
                    tds = tr.find_all('td')
                    winner = tds[2].a
                    if(len(winner.contents) == 0):
                        print(winner)
                        continue
                    
                    td = tds[1].a
                    link = td.get('href')        
                    ## read the information about the stage
                    req = session.get(pro_site + link)
                    soup = BeautifulSoup(req.content, 'lxml')
                    read_stage_results(soup, race_id, stages, riders, teams, results, i + 1)
                    
    # convert lists to dataframes
    races_df = pd.DataFrame(races)
    riders_df = pd.DataFrame(riders.values())
    teams_df = pd.DataFrame(teams.values())
    stages_df = pd.DataFrame(stages)
    results_df = pd.DataFrame(results)
    #print(riders_df.head())
    #print(stages_df.head())
    #print(results_df.head())
    races_df.to_csv('races.csv', index=False, encoding='utf-8')
    stages_df.to_csv('stages.csv', index=False, encoding='utf-8')
    riders_df.to_csv('riders.csv', index=False, encoding='utf-8')
    teams_df.to_csv('teams.csv', index=False, encoding='utf-8')
    results_df.to_csv('results.csv', index=False, encoding='utf-8') 
    pass

def scrape_tour_de_france_stages():
    stages = []
    
    # get all info from the race page
    session = requests.session()
    req = session.get(pro_site + 'race.php?id=171088&c=4')
    soup = BeautifulSoup(req.content, 'lxml')
    
    table = soup.find('table', class_='basic')
    stage_day = 0
    for tr in table.find_all('tr')[1:]:
        tds = tr.find_all('td')
        td = tds[1].a
        link = td.get('href')
        stage_day += 1
        ## read the information about the stage
        req = session.get(pro_site + link)
        stage_soup = BeautifulSoup(req.content, 'lxml')
        stage = read_stage_info(stage_soup)
        if stage is None:
            continue
        stage['day'] = stage_day
        #stage_name = stage_soup.find(string='Start/finish:').parent.next_sibling.encode('utf-8')
        stages.append(stage)
    
    stages_df = pd.DataFrame(stages)
    stages_df.to_csv('tdf_stages.txt', index=False, encoding='utf-8')
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
            name = process_string(item['Values'][2])
            value = item['Values'][16]
            if name not in riders_dict:
                riders.append({'rider_name' : name, 'cost' : value})
                riders_dict[name] = value
    
    riders_df = pd.DataFrame(riders)
    riders_df.to_csv('riders_cost.csv', index=False)
    pass

if __name__ == '__main__':
    scrape_tour_de_france_stages()
    pass