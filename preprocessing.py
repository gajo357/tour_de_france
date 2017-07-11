'''
Created on Jul 11, 2017

@author: Gajo
'''
import pandas as pd

stage_types = {'flags': 0, 
               'flat': 1,
               'hills_flat': 2,
               'hills_uphill': 3,
               'mountains_flat': 4,
               'mountains_uphill': 5
               }

def create_columns(stage_id, stages_df, races_df):
    stage_row = stages_df[stages_df['id'] == stage_id].iloc[0]
    
    day = stage_row['day']
    race_id = stage_row['race_id']
    stage_length = stage_row['length']
    stage_type = stage_row['type']
    if not stage_type:
        stage_type = 0
    stage_type = stage_types[stage_type]
    
    race_row = races_df[races_df['id'] == race_id].iloc[0]
    race_length = race_row['no_stages']
    dayratio = 1.0 * day/race_length
    race_class = race_row['class']
    
    return race_class, dayratio, stage_length, stage_type 

def preprocess_data():
    riders_df = pd.read_csv('riders.csv')
    races_df = pd.read_csv('races.csv')    
    stages_df = pd.read_csv('stages.csv')
    results_df = pd.read_csv('results.csv')
        
    #print(results_df.head())
    results_df['race_class'], results_df['race_day'], results_df['stage_length'], results_df['stage_type'] = \
        zip(*results_df['stage_id'].map(lambda stage_id: create_columns(stage_id, stages_df, races_df)))
    
    #print(results_df.head())
    
    results_df.to_csv('results_df.csv', index=False)
    
    return (riders_df, stages_df, results_df)

def get_all_rider_aliases(source_df, i):
    names = []
    surnames = []
    for name in source_df['name'][i].split():
        names.append(name.upper())
    
    for name in source_df['surname'][i].split():
        surnames.append(name.upper())
        
    link = source_df['link'][i].replace('rider/', '').split('_')
    names.append(link[0].upper())
    for name in link[1:]:
        surnames.append(name.upper())
        
    return names, surnames

def get_rider_id(row, source_df):
    names = row['rider_name'].split()
    surnames = names[1:]
    name = names[0]
          
    #print(id)
    matches = []
    best_match = 0
    for i in range(0, len(source_df['name'])):
        alnames, alsurnames = get_all_rider_aliases(source_df, i)
        count = 0
        if name in alnames:
            count += 1
        for n in surnames:
            if n in alsurnames:
                count += 2
        p = 1.0 * count/len(names)
        if p <= 0.5:
            continue
        if p > best_match:
            matches = []
            matches.append(source_df['id'][i])
            best_match = p 
        elif p == best_match:
            matches.append(source_df['id'][i])
    
    if len(matches) == 0:
        return 0
               
    return matches[0]

def match_riders_with_costs():
    riders_df = pd.read_csv('riders.csv')
    riders_cost_df = pd.read_csv('riders_cost.csv')
    riders_cost_df['id'] = riders_cost_df.apply(lambda row: get_rider_id(row, riders_df), axis=1)
    
    riders_cost_df.to_csv('riders_w_cost.csv', index=False)

def transform_tdf_stages():
    df = pd.read_csv('tdf_stages.txt')
    
    df['stage_type'] = df['type'].apply(lambda x: stage_types[x])
    df['stage_length'] = df['length']
    
    df['race_class'] = df['type'].apply(lambda x: 2)
    no_stages = len(df['day'])
    df['race_day'] = df['day'].apply(lambda x: 1.0*x/no_stages)
    
    df = df[['stage_type', 'stage_length', 'race_class', 'race_day']]

    df.to_csv('tdf_stages.csv', index=0)
    pass

if __name__ == '__main__':
    transform_tdf_stages()