# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:14:07 2020

@author: hazar
"""

# In[1]:
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
spark = SparkSession.builder.appName('collaborative filtering').getOrCreate()
       
# Loads data
ratings = pd.read_csv("C:/Users/hazar/Documents/MSIST/MSIST S20/IOT Managment/project/combined_data_1.txt/combined_data_1.txt", header = None, names = ['Customer_id', 'Rating', 'timestamp'], usecols = [0,1,2])

print(ratings.iloc[::1, :])
"""
         Customer_id  Rating   timestamp
0                 1:     NaN         NaN
1            1488844     3.0  2005-09-06
2             822109     5.0  2005-05-13
3             885013     4.0  2005-10-19
4              30878     4.0  2005-12-26
             ...     ...         ...
24058258     2591364     2.0  2005-02-16
24058259     1791000     2.0  2005-02-10
24058260      512536     5.0  2005-07-27
24058261      988963     3.0  2005-12-20
24058262     1704416     3.0  2004-06-02

[24058263 rows x 3 columns]
"""
ratings.count()
"""
Customer_id    24058263
Rating         24053764
timestamp      24053764
dtype: int64
"""


# In[]:
## Start Exploring The dataframe 
# --------------------------------------------------------------------------------------------------------------------
# How the rating is spread 
p = ratings.groupby('Rating')['Rating'].agg(['count'])
print (p)
"""
          count
Rating         
1.0     1118186
2.0     2439073
3.0     6904181
4.0     8085741
5.0     5506583
"""

p.plot(kind = 'bar', figsize=(20,20))
""" We Notice that the majority of the movies were rated 4 """

# get How many movies were rated
movie_count = ratings.isnull().sum()[1]
""" 4499 movies with the dataframe I have """

# get how many cusotmer we have 
customers_count = ratings['Customer_id'].nunique() - movie_count
""" 470758 Customer who rated the movies in this dataframe"""

# get rating count
rating_count = ratings['Customer_id'].count() - movie_count
""" 24053764 ratings """

# get Avergae rating per Cutomer and the avergae rating per movie 
rating_count/customers_count 
""" Each customer rated 51 movies """
rating_count/movie_count 
""" Each movie was rated by 5346 customer """ 

# In[]
# Collect the movie_id and add it as a column to the end of the data set 
# The movie Id is a record in the dataset with Null other features 
ratings_nan = ratings.isnull()
ratings_nan = ratings_nan[ratings_nan['Rating'] == True]
ratings_nan = ratings_nan.reset_index()
ratings_nan = ratings_nan[['index','Rating']]

"""
         index  Rating
0            0    True
1          548    True
2          694    True
3         2707    True
4         2850    True
       ...     ...
4494  24046714    True
4495  24047329    True
4496  24056849    True
4497  24057564    True
4498  24057834    True

[4499 rows x 2 columns]
"""

# In[] Clean the Data
#--------------------------------------------------------------------------------------------------------------------------------
movies = []
movie_id = 1
for i,j in zip(ratings_nan['index'][1:],ratings_nan['index'][:-1]):
    # numpy array
    temp = np.full((1,i-j-1), movie_id)
    movies = np.append(movies, temp)
    movie_id += 1


last_record = np.full((1,len(ratings) - ratings_nan.iloc[-1, 0] - 1),movie_id)
movies = np.append(movies, last_record)

ratings= ratings[pd.notnull(ratings['Rating'])]

# add the Movie_id column to the ratings data set 
ratings['movie_id'] = movies.astype(int)
ratings['Customer_id'] = ratings['Customer_id'].astype(int)

print(ratings)

"""
          Customer_id  Rating   timestamp  movie_id
1             1488844     3.0  2005-09-06         1
2              822109     5.0  2005-05-13         1
3              885013     4.0  2005-10-19         1
4               30878     4.0  2005-12-26         1
5              823519     3.0  2004-05-03         1
              ...     ...         ...       ...
24058258      2591364     2.0  2005-02-16      4499
24058259      1791000     2.0  2005-02-10      4499
24058260       512536     5.0  2005-07-27      4499
24058261       988963     3.0  2005-12-20      4499
24058262      1704416     3.0  2004-06-02      4499

[24053764 rows x 4 columns]
"""

# In[]
# Start Exploring Movie Titles with Spark 
#-------------------------------------------------------------------------------------------------------------------------------------
# Read the movie titles 
movies_titles2 = pd.read_csv("C:/Users/hazar/Documents/MSIST/MSIST S20/IOT Managment/project/movie_titles.csv", encoding = "ISO-8859-1", header = None, names = ['movie_id', 'year', 'title'], usecols = [0,1,2])
print(movies_titles2)
"""
       movie_id    year                                              title
0             1  2003.0                                    Dinosaur Planet
1             2  2004.0                         Isle of Man TT 2004 Review
2             3  1997.0                                          Character
3             4  1994.0                       Paula Abdul's Get Up & Dance
4             5  2004.0                           The Rise and Fall of ECW
        ...     ...                                                ...
17765     17766  2002.0  Where the Wild Things Are and Other Maurice Se...
17766     17767  2004.0                  Fidel Castro: American Experience
17767     17768  2000.0                                              Epoch
17768     17769  2003.0                                        The Company
17769     17770  2003.0                                       Alien Hunter

[17770 rows x 3 columns]
"""
movies_titles2.count()
"""
movie_id    17770
year        17763
title       17770
dtype: int64
"""

# In[3]
# How many videos for each production year
#------------------------------------------------------------------------------------------------------------------------
Summary_titles= movies_titles2.groupby (['year']).count()
print(Summary_titles)

"""
        movie_id  title
year                   
1896.0         1      1
1909.0         1      1
1914.0         2      2
1915.0         5      5
1916.0         4      4
         ...    ...
2001.0      1184   1184
2002.0      1310   1310
2003.0      1271   1271
2004.0      1436   1436
2005.0       512    512

[94 rows x 2 columns]
"""
plt= Summary_titles.plot(kind = 'bar', figsize=(20,20))

# In[]
# connect the rating file with the movie title file
#------------------------------------------------------------------------------------------------------------------------
combined_data= ratings.join(movies_titles2.set_index('movie_id'), on='movie_id')
"""
          Customer_id  Rating   timestamp  movie_id    year            title
1             1488844     3.0  2005-09-06         1  2003.0  Dinosaur Planet
2              822109     5.0  2005-05-13         1  2003.0  Dinosaur Planet
3              885013     4.0  2005-10-19         1  2003.0  Dinosaur Planet
4               30878     4.0  2005-12-26         1  2003.0  Dinosaur Planet
5              823519     3.0  2004-05-03         1  2003.0  Dinosaur Planet
              ...     ...         ...       ...     ...              ...
24058258      2591364     2.0  2005-02-16      4499  2002.0       In My Skin
24058259      1791000     2.0  2005-02-10      4499  2002.0       In My Skin
24058260       512536     5.0  2005-07-27      4499  2002.0       In My Skin
24058261       988963     3.0  2005-12-20      4499  2002.0       In My Skin
24058262      1704416     3.0  2004-06-02      4499  2002.0       In My Skin

[24053764 rows x 6 columns]
"""

# In[]
# Exploring the combined data to see if there is any signinfcant impact between the year f the movie and its rating
# ----------------------------------------------------------------------------------------------------------------------
av= combined_data.groupby('year').mean()['Rating']
print(av)
"""
year
1915.0    3.314961
1916.0    3.544333
1917.0    3.311594
1918.0    3.230769
1920.0    3.377354
  
2001.0    3.526722
2002.0    3.456883
2003.0    3.613465
2004.0    3.618105
2005.0    3.659420
Name: Rating, Length: 89, dtype: float64
+----+------------------+
"""
av.plot(figsize=(20,20))
 
"""We notice:  There is no significant differance in the rating between old movies or new ones. """

# In[]


