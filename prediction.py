'''
Created on Jul 1, 2017

@author: Gajo
'''
import pandas as pd
import numpy as np
from sklearn.externals import joblib
 

def read_stages():
    tdf_stages = pd.read_csv('tdf_stages.txt', header=None, names=['stage_name', 'stage_length', 'stage_type'])
    
    return tdf_stages
    
def predict_results():
    stages = read_stages()
    riders_df = pd.read_csv('riders_w_cost.csv')
    
    # load the model
    reg = joblib.load('model.pkl')
    
    p_by_rider = {}
    for i in range(2, len(stages['stage_name'])):
        stage = stages.iloc[i]
        values = []
        for j in range(0, len(riders_df['id'])):
            rider = riders_df.iloc[j]
            values.append([rider['id'], rider['height'], rider['weight'], rider['one_day'], rider['gc'], rider['time_trial'], rider['sprint'], rider['pcs_rank'], rider['uci_rank'], stage['stage_length'], stage['stage_type']])
        prediction = reg.predict(values)
        for j in range(0, len(riders_df['id'])):
            if j not in p_by_rider:
                p_by_rider[j] = []
            p_by_rider[j].append(prediction[j])
    
    values = []
    for j in range(0, len(riders_df['id'])):
        rider = riders_df.iloc[j]
        all_points = np.mean(p_by_rider[j])
        cost = rider['cost']
        values.append([rider['rider_name'], all_points, cost, 1.0/(cost/1000000*all_points/100)])
    
    df = pd.DataFrame(values, columns=['rider_name', 'points', 'cost', 'value_for_money'])
    df.sort_values('value_for_money', axis=0, ascending=False, inplace=True)
    print(df)
    df.to_csv('pred.csv', index=False)
    
if __name__ == '__main__':
    predict_results()