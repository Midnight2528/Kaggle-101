import pandas as pd
leak_df = pd.read_feather('leak.feather')
leak_df = leak_df[leak_df['building_id'] != 13].reset_index(drop=True)
leak_df = leak_df[leak_df['building_id'] != 14].reset_index(drop=True)
leak_df.to_feather('leak.feather')
