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
