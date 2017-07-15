'''
Created on Jul 1, 2017

@author: Gajo
'''
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from training_model import features
import os.path

def read_stages():
    tdf_stages = pd.read_csv('tdf_stages.csv')
    tdf_stages['season'] = tdf_stages['race_class'].apply(lambda x: 2017)
    tdf_stages['race_length'] = tdf_stages['race_day'].apply(lambda x: 21)
    tdf_stages['race_day'] = tdf_stages['race_day'].apply(lambda x: int(21*x))
    
    return tdf_stages.as_matrix(features)
    
def predict_results():
    stages = read_stages()
    riders_df = pd.read_csv('riders_w_cost.csv')
    
    p_by_rider = {}
    p_by_stage = {}
    # predict rider by rider
    values = []
    for i in range(0, len(riders_df['id'])):
        rider = riders_df.iloc[i]
        rider_id = rider['id']
        rider_cost = rider['cost']
        rider_name = rider['rider_name']
            
        # load the model
        file_path = 'models/model_{}.pkl'.format(rider_id)
        if not os.path.exists(file_path):
            continue
        reg = joblib.load(file_path)
        
        # collect all stages infos, so we could predict all at once
        predictions = reg.predict(stages)
        p_by_rider[rider_id] = predictions
        for j in range(0, len(predictions)):
            if j not in p_by_stage:
                p_by_stage[j] = []
            p_by_stage[j].append((rider_name, predictions[j]))
    
        all_points = np.sum(predictions)/(60*60) # avg time in hours
        values.append([rider_name, all_points, rider_cost, 1.0/(rider_cost/1000000*all_points)])
    
    with open("Output.csv", "w") as text_file:
        for s, r in p_by_stage.items():
            text_file.write("\n")
            sorted_by_result = sorted(r, key=lambda x: x[1])
            text_file.write("Stage: {0}\n".format(s))
            for rinfo in sorted_by_result:
                text_file.write("{0},{1}\n".format(rinfo[0], rinfo[1]))
                
        
    #results = pd.DataFrame(p_by_stage, orient='index')
    #results.to_csv('pred_by_stage.csv')
    
    df = pd.DataFrame(values, columns=['rider_name', 'points', 'cost', 'value_for_money'])
    df.sort_values('value_for_money', axis=0, ascending=False, inplace=True)
    print(df)
    df.to_csv('pred.csv', index=False)
    
    df.sort_values('points', axis=0, ascending=True, inplace=True)
    df.to_csv('pred_by_time.csv', index=False)
    
if __name__ == '__main__':
    predict_results()