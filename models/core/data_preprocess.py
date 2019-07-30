import json
import pandas as pd
import _pickle as pickle    

def process_nightscout_json(json_name,save_name):
    entries_json = json.load(open(json_name, 'r'))
    entries_df = pd.DataFrame(entries_json)
    entries_df['datetime'] = pd.to_datetime(entries_df['dateString'])
    entries_df.sort_values(by=['datetime'], ascending=True, inplace=True)
    pickle.dump(entries_df, open(save_name, 'wb'))

if __name__ == "__main__":
    process_nightscout_json('data/28176124_entries__to_2018-11-05.json', 'data/28176124_entries__to_2018-11-05_df.pkl')
    