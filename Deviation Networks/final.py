import pandas as pd
df = pd.read_csv ('/content/synthetic_aws5f5533_final_data.csv')

#anamoly = [61, 76, 69, 72, 74, 72, 58, 61, 68, 62.056]

def categorize(row):
  if row['Value'] in [63, 63, 75, 69, 59, 80, 75, 76, 77, 74, 62.056]:
    print ('Yes')
    return 1
  else:
    return 0

df['ana'] = df.apply(lambda row: categorize(row), axis=1)

#df = df.drop('Date',axis=1)
df.to_csv('final.csv')