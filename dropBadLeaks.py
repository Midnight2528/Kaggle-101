leak_df = pd.read_feather('kaggle/input/leak.feather')

leak_df = leak_df [ leak_df['building_id'] != 13 ]
leak_df = leak_df [ leak_df['building_id'] != 14 ]

leak_df.to_feather('leak.feather')
