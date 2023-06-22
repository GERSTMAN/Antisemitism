# Antisemitism
The goal of this project is to obtain insight on the prevalance of Anti-Semitic activity in the United States, based on data collected by the ADL for their H.E.A.T. map project.

```
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
```


# The Data

1. Upload Antisemitism data from the Anti defamation league Data base

The following data was obtained from https://www.adl.org/resources/tools-to-track-hate/heat-map

```
url = "https://raw.githubusercontent.com/GERSTMAN/Antisemitism/main/extremism%20(3).csv"
antisemitism = pd.read_csv(url,on_bad_lines='skip')
```

A glimpse of the raw data:

```
antisemitism.sample(3)
```

|	|id|	date|	city|	state|	type|	ideology|	subideology|	group|	description|	image|	year|	month|	day|	year.1|	month.1|	day.1|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|9099|	34199|	2/1/2022|	Denton|	TX|	White Supremacist Propaganda|	Right Wing (White Supremacist)|	NaN|	Patriot Front|	Patriot Front, a white supremacist group, dist...|	NaN|	2022|	2|	1|	2022|	2|	1|
|5125|	36907|	6/1/2022|	Spencer|	MA|	White Supremacist Propaganda|	Right Wing (White Supremacist)|	NaN|	Patriot Front|	Patriot Front, a white supremacist group, dist...|	NaN|	2022|	6|	1|	2022|	6|	1|
|35062|	1392|	Jul-16|	Los Angeles|	CA|	Antisemitic Incident:Vandalism|	NaN|	NaN|	NaN|	Valuable object donated to the University of S...|	NaN|	2016|	7|	1|	2016|	7|	1|
```
antisemitism.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 36051 entries, 0 to 36050
Data columns (total 16 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   id           36051 non-null  int64  
 1   date         36051 non-null  object 
 2   city         36051 non-null  object 
 3   state        36051 non-null  object 
 4   type         36051 non-null  object 
 5   ideology     22459 non-null  object 
 6   subideology  0 non-null      float64
 7   group        21762 non-null  object 
 8   description  36050 non-null  object 
 9   image        8043 non-null   object 
 10  year         36051 non-null  int64  
 11  month        36051 non-null  int64  
 12  day          36051 non-null  int64  
 13  year.1       36051 non-null  int64  
 14  month.1      36051 non-null  int64  
 15  day.1        36051 non-null  int64  
dtypes: float64(1), int64(7), object(8)
memory usage: 4.4+ MB
```

# Data Cleanup

Year, Month and Day are columns that I constructed from the Date column, to avoid data loss due to inconsistent date formmating.
Turns out this saved most of the data from 2021.
year.1, month.1 and day.1 are columns where I dropped copies of the above values for backup.
Eventually that wasn't needed, and those columns were the first to be cleared from the df.
The subideology category wasn't put to use at all, and therefore parsed out.

```
antisemitism = antisemitism.drop(columns=['year.1','month.1','day.1','subideology'])

antisemitism['year'] = antisemitism['year'].astype('int16')
antisemitism['month'] = antisemitism['month'].astype('int16')
antisemitism['day'] = antisemitism['day'].astype('category')

antisemitism.sample(1)
```
|	|id|	date|	city|	state|	type|	ideology|	group|	description|	image|	year|	month|	day|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|5530|	35439|	5/1/2022|	Williamsburg|	VA|	White Supremacist Propaganda|	Right Wing (White Supremacist)|	White Lives Matter|	Individuals associated with White Lives Matter...|	NaN|	2022|	5|	1.0|

```
antisemitism.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 36051 entries, 0 to 36050
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   id           36051 non-null  int64   
 1   date         36051 non-null  object  
 2   city         36051 non-null  object  
 3   state        36051 non-null  object  
 4   type         36051 non-null  object  
 5   ideology     22459 non-null  object  
 6   group        21762 non-null  object  
 7   description  36050 non-null  object  
 8   image        8043 non-null   object  
 9   year         36051 non-null  int16   
 10  month        36051 non-null  int16   
 11  day          20798 non-null  category
dtypes: category(1), int16(2), int64(1), object(8)
memory usage: 2.6+ MB

antisemitism.set_index('id')

counts = antisemitism.nunique()

counts
id             36051
date            2096
city            5837
state             51
type              20
ideology           7
group            325
description    27381
image            199
year              21
month             12
day               31
dtype: int64
```


Columns with relatively few unique values will be converted to Categorical

```
antisemitism['state'] = antisemitism['state'].astype('category')
antisemitism['type'] = antisemitism['type'].astype('category')
antisemitism['ideology'] = antisemitism['ideology'].astype('category')
antisemitism['group'] = antisemitism['group'].astype('category')

#Courtesy of chatGPT
antisemitism.loc[antisemitism['image'].notnull(), 'image'] = 'image provided'

antisemitism['image'] = antisemitism['image'].astype('category')
```
Added a reconstructed date field 'fdate' to facilitate further analysis with a 'proper' datetime field.

```
antisemitism['fdate'] = pd.to_datetime(antisemitism['year'].astype(str) + '-' + antisemitism['month'].astype(str) + '-' + (antisemitism['day'].fillna(1).astype(int)).astype(str))

antisemitism.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 36051 entries, 0 to 36050
Data columns (total 13 columns):
 #   Column       Non-Null Count  Dtype         
---  ------       --------------  -----         
 0   id           36051 non-null  int64         
 1   date         36051 non-null  object        
 2   city         36051 non-null  object        
 3   state        36051 non-null  category      
 4   type         36051 non-null  category      
 5   ideology     22459 non-null  category      
 6   group        21762 non-null  category      
 7   description  36050 non-null  object        
 8   image        8043 non-null   category      
 9   year         36051 non-null  int16         
 10  month        36051 non-null  int16         
 11  day          20798 non-null  category      
 12  fdate        36051 non-null  datetime64[ns]
dtypes: category(6), datetime64[ns](1), int16(2), int64(1), object(3)
memory usage: 1.8+ MB
```

After reducing the the dataframe to 40% of it's original size, we can now drill down into the data:



# the Demographic Factor

```
#State name dictionary. courtesy of https://gist.github.com/rogerallen/1583593. Then I added a few

 state_ab_dict = {
"Alabama": "AL",
"Alaska": "AK",
"Arizona": "AZ",
"Arkansas": "AR",
"California": "CA",
"Colorado": "CO",
"Connecticut": "CT",
"Delaware": "DE",
"Florida": "FL",
"Georgia": "GA",
"Hawaii": "HI",
"Idaho": "ID",
"Illinois": "IL",
"Indiana": "IN",
"Iowa": "IA",
"Kansas": "KS",
"Kentucky": "KY",
"Louisiana": "LA",
"Maine": "ME",
"Maryland": "MD",
"Massachusetts": "MA",
"Michigan": "MI",
"Minnesota": "MN",
"Mississippi": "MS",
"Missouri": "MO",
"Montana": "MT",
"Nebraska": "NE",
"Nevada": "NV",
"New Hampshire": "NH",
"New Jersey": "NJ",
"New Mexico": "NM",
"New York": "NY",
"North Carolina": "NC",
"North Dakota": "ND",
"Ohio": "OH",
"Oklahoma": "OK",
"Oregon": "OR",
"Pennsylvania": "PA",
"Rhode Island": "RI",
"South Carolina": "SC",
"South Dakota": "SD",
"Tennessee": "TN",
"Texas": "TX",
"Utah": "UT",
"Vermont": "VT",
"Virginia": "VA",
"Washington": "WA",
"West Virginia": "WV",
"Wisconsin": "WI",
"Wyoming": "WY",
"District of Columbia": "DC",
"Alab.": "AL",
"Ark.": "AR",
"Calif.": "CA",
"Colo.": "CO",
"Conn.": "CT",
"Del.": "DE",
"D.C.": "DC",
"Ky.": "KY",
"La.": "LA",
"Maine †": "ME",
"Md.": "MD",
"Mass.": "MA",
"Mich.": "MI",
"Minn.": "MN",
"Miss.": "MS",
"Mo.": "MO",
"Mont.": "MT",
"Neb. †": "NE",
"Nev.[q]": "NV",
"N.H.": "NH",
"N.J.[r]": "NJ",
"N.M.": "NM",
"N.Y.": "NY",
"N.C.": "NC",
"N.D.": "ND",
"Okla.": "OK",
"Pa.": "PA",
"R.I.": "RI",
"S.C.": "SC",
"S.D.": "SD",
"Tenn.": "TN",
"Texas[s]": "TX",
"Vt.": "VT",
"Va.": "VA",
"Wash.": "WA",
"W.Va.": "WV",
"Wis.": "WI",
"Wyo.": "WY"}
```

1. Jewish population

```
url_1 = 'https://en.wikipedia.org/w/index.php?title=American_Jews&oldid=1160034360'
```

Here and later on I used the permanent link to the most recent version in the entry's history so that future edits won't disrupt the project

```
dfs_1 = pd.read_html(url_1)

jews_num = dfs_1[11]
```

2. Muslim population

```
url_2 = 'https://en.wikipedia.org/w/index.php?title=Islam_in_the_United_States&oldid=1160898319'

dfs_2 = pd.read_html(url_2)

isl_num = dfs_2[2]
```

3. General population

```
url_3 = 'https://en.wikipedia.org/w/index.php?title=2020_United_States_census&oldid=1161035383'

dfs_3 = pd.read_html(url_3)

tot_num = dfs_3[2]
```

4. Votes

```
url_4 = 'https://en.wikipedia.org/w/index.php?title=2020_United_States_presidential_election&oldid=1160932277'

dfs_4 = pd.read_html(url_4)

voters = dfs_4[21]

votepct = voters.iloc[:,[0,1,4]]
```
The dataframe initially was a multi index, which would've complicated matters when attempting to merge the data into a single dataframe.

```
votepct_f = votepct.copy()
votepct_f.columns = votepct.columns.get_level_values(0)
```

5. Merge tables

```
jews_num['state']= jews_num['States and territories'].apply(lambda x: state_ab_dict.get(x, 'Unknown'))

isl_num['state']= isl_num['State'].apply(lambda x: state_ab_dict.get(x, 'Unknown'))

tot_num['state']= tot_num['State'].apply(lambda x: state_ab_dict.get(x, 'Unknown'))

votepct_f['state']= votepct_f['State or district'].apply(lambda x: state_ab_dict.get(x, 'Unknown'))

jews_num = jews_num.rename(columns={'American Jews (2020)[68]': 'Jewish Population'})
isl_num = isl_num.rename(columns={'Muslim (estimate)[138]': 'Muslim Population'})
tot_num = tot_num.rename(columns={'Population as of 2020 census[80]': 'Total Population'})
votepct_f = votepct_f.rename(columns={'Biden/Harris Democratic': 'Biden votes',
                                    'Trump/Pence Republican': 'Trump votes'})

demographics = pd.merge(jews_num[['state', 'Jewish Population']],
                     isl_num[['state', 'Muslim Population']],
                     on='state')
demographics = pd.merge(demographics, tot_num[['state', 'Total Population']], on='state')
demographics = pd.merge(demographics, votepct_f[['state', 'Biden votes', 'Trump votes']], on='state')

demographics
```

|	|state|	Jewish Population|	Muslim Population|	Total Population|	Biden votes|	Trump votes|
|---|---|---|---|---|---|---|
|0|	AL|	10325|	23550|	5024279|	849624|	1441170|
|1|	AK|	5750|	400|	733391|	153778|	189951|
|2|	AZ|	106300|	109765|	7151502|	1672143|	1661686|
...
|48|	WV|	2310|	849|	1793716|	235984|	545382|
|49|	WI|	33455|	68699|	5893718|	1630866|	1610184|
|50|	WY|	1150|	226|	576851|	73491|	193559|

5. Insert Antisemitism Data

```
anti_20 = antisemitism.loc[antisemitism['year']==2020]

anti_20
```

```
twenty_state = pd.pivot_table(anti_20, values='id', index='state',
               aggfunc='count')

twenty_state['antisemitic'] = pd.pivot_table(anti_20[anti_20['type'].str.contains('antisemitic', case = False)], values='id', index='state',
               aggfunc='count')

twenty_state['white supremacist'] = pd.pivot_table(anti_20[anti_20['type'].str.contains('white supremacist', case = False)], values='id', index='state', aggfunc='count')

twenty_state['police shootout'] = pd.pivot_table(anti_20[anti_20['type'].str.contains('police shootout', case = False)], values='id', index='state', aggfunc='count')

twenty_state['murder'] = pd.pivot_table(anti_20[anti_20['type'].str.contains('murder', case = False)], values='id', index='state', aggfunc='count')

twenty_state['terrorist'] = pd.pivot_table(anti_20[anti_20['type'].str.contains('terrorist', case = False)], values='id', index='state', aggfunc='count')

twenty_state
```
