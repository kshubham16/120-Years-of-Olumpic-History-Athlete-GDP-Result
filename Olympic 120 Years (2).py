#!/usr/bin/env python
# coding: utf-8

# # 120 Years of Olympics History: Athlete, GDP, Result
# ## 1896 - 2016

# In[578]:


from IPython.display import Image
Image(filename = "ol_img.jpg", width=1000)


# In[2]:


get_ipython().system('pip install xgboost')


# In[1]:


# Import all required libraries

import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
#from xgboost.xgbclassifier import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[2]:


# Upload Athlete dataset
dfa = pd.read_excel('athlete_events.xlsx')
dfa


# #### Content
# - The file athlete_events.csv contains 271116 rows and 15 columns. Each row corresponds to an individual athlete competing in an individual Olympic event (athlete-events). The columns are:
# 
# - ID - Unique number for each athlete
# - Name - Athlete's name
# - Sex - Male(M) or Female(F)
# - Age - Integer
# - Height - In centimeters
# - Weight - In kilograms
# - Team - Team name
# - NOC - National Olympic Committee 3-letter code
# - Games - Year and season
# - Year - Integer
# - Season - Summer or Winter
# - City - Host city
# - Sport - Sport
# - Event - Event
# - Medal - Gold, Silver, Bronze, or NA

# In[3]:


dfa.shape # Shape of Athlete dataset(Rows, Columns)


# In[4]:


# Upload Region Dataset

dfr = pd.read_csv('noc_regions.csv')
dfr.drop('notes', axis=1, inplace=True)
dfr.head(5)


# In[5]:


# Merge Region into Athlete Dataset

df = dfa.merge(dfr, how='left',on='NOC')
df


# In[6]:


# Drop Unessential Columns
df.drop(['ID', 'Team', 'Games'], axis=1, inplace=True)
df


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


# Save new dataset raw dataset
#df.to_csv('ath_data.csv')


# In[10]:


print("Age of the Participants :\n\n", df.Age.sort_values(ascending=True).unique())


# In[11]:


print("Height of the Participants : \n\n", df.Height.sort_values(ascending=True).unique())


# In[12]:


print("Weigh of the Participants: \n\n", df.Weight.sort_values(ascending=True).unique())


# In[13]:


print("Olympics happend in these years :\n\n",df.Year.sort_values(ascending=True).unique())


# In[14]:


# Count How many null values in each columns
df.isnull().sum()


# In[15]:


# Viewing missing value with help of heatmap
plt.figure(figsize=(20,8))
sns.heatmap(df.isnull())
plt.show()


# ## Data Cleaning

# ### Cleanning Data
# 
# - Change name of some countries and others
# - Age contains null values, Fill with mean/median.
# - Height contains nulll values, Fill with mean/median.
# - Weight contains null values, Fill with mean/median.
# - Medal has null values, Fill with No Medal.

# #### Naming Conversion

# In[16]:


#df[df.region=='UK']
#df[df.region=='Trinidad']
df.region.replace({"UK":'United Kingdom', "Trinidad":'Trinidad and Tobago'},inplace=True)


# In[17]:


df.rename(columns={'Sex':'Gender'},inplace=True)


# In[18]:


df.rename(columns={'region':'Region'},inplace=True)


# #### Drop Duplicates

# In[19]:


# df.duplicated().sum()
# df.drop_duplicates(keep='last', inplace=True, ignore_index=True )
dfa.duplicated(subset=['Name', 'NOC', 'Season', 'City', 'Sport', 'Event', 'Medal']).sum()


# In[20]:


# Drop duplicates values on the basis of some columns.
df = df.drop_duplicates(subset=['Name', 'NOC', 'Season', 'City', 'Sport', 'Event', 'Medal'])


# In[21]:


# We have a new shaped dataset
df.shape


# In[22]:


df.isnull().sum()


# #### Age

# In[23]:


# df.Age.isnull().sum()


# In[24]:


df['Age'].mean() #mean of Age column
df.Age.fillna(df['Age'].mean(), inplace=True) #fill with mean values and change parmanently  
df.Age.isnull().sum() # Check have there any many null values or not. 
df['Age'] = df['Age'].astype(int) # Change datatype into integer.


# #### Height

# In[25]:


h = df.Height.mean()
h


# In[26]:


m = df.Height.median()
m


# In[27]:


df.Height.fillna(h, inplace=True)


# In[28]:


df.Height.isnull().sum()


# In[29]:


df['Height'] = df['Height'].astype(int)


# #### Weight

# In[30]:


# df.Weight.isnull().sum()


# In[31]:


w_md = df.Weight.median()
w_md


# In[32]:


w_mn = df.Weight.mean()
w_mn


# In[33]:


df.Weight.mean()
df.Weight.fillna(df.Weight.mean(), inplace=True)
df['Weight'] = df['Weight'].astype(int)
df.isnull().sum()


# In[34]:


display(df.dtypes)


# In[35]:


df.shape


# In[36]:


# Save cleaned dataset(except medal) in our PC
# cl_df = df
# cl_df = cl_df.dropna()
# cl_df.isnull().sum()
# cl_df.to_csv("clnRowDt.csv")


# In[37]:


# cl_df.shape


# ## Visualization

# ## 1. Gender wise distribution

# ### 1. Total Participants

# In[38]:


# Find how many male and female participants were in olympics game
# Here, I have work on raw dataset
gender = df['Gender'].value_counts()
print('Total Male(M) and Female(F) paricipants :- ')
print(gender)


# In[39]:


plt.figure(figsize = (16,9)) # figure size with ratio 16:9
sns.set(style='darkgrid',) # background darkgrid style of graph 
plt.show()
sns.countplot(df.Gender)
plt.title('Total Participants \n (M = Male || F = Female) \n')


# In[40]:


# Find how %age of both partcipants
plt.figure(figsize=(12,8))
plt.title('Gender Distribution')
plt.pie(gender, labels=gender.index, autopct='%1.1f%%', startangle=90, shadow=True);


# ### 2. Male and Female, Who won any medal

# ### Female

# In[41]:


# We found how many Gold, Silver, Bronze Medals won by women?
female = df[df.Gender == 'F']
female=female.dropna()
f = female.Medal.value_counts()
print(f)
plt.figure(figsize=(12,8))
plt.title('Won By Female Participants')
plt.pie(f, labels=f.index, autopct='%1.1f%%', startangle=90, shadow=True);


# ### Male

# In[42]:


# We found how many Gold, Silver, Bronze Medals won by women?

male = df[df.Gender == 'M']
male=male.dropna()
m = male.Medal.value_counts()
print(m)
plt.figure(figsize=(12,8))
plt.title('Won By Male Participants')
plt.pie(m, labels=m.index, autopct='%1.1f%%', startangle=90, shadow=True);


# ## 2. Country wise distribution

# ### 1. Top 50 participants countries

# In[43]:


#Top 50 countries participating
t50_countries = df.Region.value_counts().sort_values(ascending=False).head(50)
t50_countries
# df.shape


# In[44]:


# 18603 times participant represented USA. One person play many format in his/her Sports in particular Year.


# In[45]:


#plot for the top 50 countries

plt.figure(figsize=(30,25))
plt.xticks(rotation=90)
plt.title('Overall participation by Country')
plt.xlabel('Countries', fontsize=15)
sns.barplot(x=t50_countries, y=t50_countries.index, palette = 'rocket');


# In[46]:


# Drop all null values and stored in a new varable called dff.


# In[47]:


dff = df.dropna()
dff.reset_index(drop=True, inplace=True)


# In[48]:


# dff.isnull().sum()
# dff.shape


# In[49]:


d1 = dff[['Region', 'Medal']]
d1.head(3)


# In[50]:


x = d1[d1.Medal == 'Gold'].Region.value_counts().sort_values(ascending=False).head(50)
print("Top 50 countries won Gold Medal.\n",x.head())
y = d1[d1.Medal == 'Silver'].Region.value_counts().sort_values(ascending=False).head(50)
print("\nTop 50 countries won Silver Medal.\n",y.head())
z = d1[d1.Medal == 'Bronze'].Region.value_counts().sort_values(ascending=False).head(50)
print("\nTop 50 countries won Bronze Medal.\n",z.head())


# In[51]:


# Shown how many 
plt.figure(figsize=(20,15))
plt.xticks(rotation=90)

plt.subplot(1,3,1)
plt.title('Won Gold By Top 50 Countries', fontsize=15)
#plt.xlabel('Countries', fontsize=15)
plt.ylabel('Gold Medals', fontsize=15)
sns.barplot(x=x, y=x.index, palette = 'rocket');
#sns.lineplot(x=x, y=x.index, palette = 'crest');

plt.subplot(1,3,2)
plt.title('Won Silver By Top 50 Countries', fontsize=15)
#plt.xlabel('Countries', fontsize=15)
plt.ylabel('Silver Medals', fontsize=15)
sns.barplot(x=y, y=y.index, palette = 'rocket');
#sns.lineplot(x=y, y=y.index, color='b', palette = 'crest');

plt.subplot(1,3,3)
plt.title('Won Silver By Top 50 Countries', fontsize=15)
plt.xlabel('Countries', fontsize=15)
plt.ylabel('Bronze Medals', fontsize=15)
sns.barplot(x=z, y=z.index, palette = 'rocket');
#sns.lineplot(x=z, y=z.index, color='b', palette = 'crest');

plt.tight_layout()


# ## 3. Year Wise Medal Distribution

# In[52]:


dff.head()


# In[53]:


yr = dff[['Year', 'Medal']]
yr.Medal.value_counts()


# In[54]:


yr.Year.value_counts()


# In[55]:


# Medals won by each year
plt.figure(figsize=(20,15))
plt.xticks(rotation=90)

plt.subplot(3,1,1)
plt.title('Year wise distribution for Medals', fontsize=15)
plt.xlabel('Years', fontsize=15)
plt.ylabel('No. of Medals', fontsize=15)
sns.countplot(x=yr.Year, palette = 'Spectral');
# sns.lineplot(x=g.index, y=g, palette = 'rocket');


# In[56]:


g = yr[yr.Medal == 'Gold'].Year.value_counts().sort_index(ascending=True).head(50)
print('Won Bronze Medal in each year: \n', g.head())
s = yr[yr.Medal == 'Silver'].Year.value_counts().sort_index(ascending=True).head(50)
print('\nWon Bronze Medal in each year: \n', s.head())
b = yr[yr.Medal == 'Bronze'].Year.value_counts().sort_index(ascending=True).head(50)
print('\nWon Bronze Medal in each year: \n',b.head(5))


# In[57]:


plt.figure(figsize=(20,15))
plt.xticks(rotation=90)

plt.subplot(3,1,1)
plt.title('Year wise distribution for Gold Medals', fontsize=15)
#plt.xlabel('Countries', fontsize=15)
plt.ylabel('Gold Medals', fontsize=15)
sns.barplot(x=g.index, y=g, palette = 'Spectral');
# sns.lineplot(x=g.index, y=g, palette = 'rocket');

plt.subplot(3,1,2)
plt.title('Year wise distribution for Silver Medals', fontsize=15)
#plt.xlabel('Countries', fontsize=15)
plt.ylabel('Silver Medals', fontsize=15)
sns.barplot(x=s.index, y=s, palette = 'Spectral');

plt.subplot(3,1,3)
plt.title('Year wise distribution for Bronze Medals', fontsize=15)
plt.xlabel('Countries', fontsize=15)
plt.ylabel('Bronze Medals', fontsize=15)
sns.barplot(x=b.index, y=b, palette = 'Spectral');

plt.tight_layout()


# In[58]:


# Compare all three medals in each year

get_ipython().run_line_magic('matplotlib', 'inline')
# plt.figure(figsize=(20,30))
# plt.xticks(rotation=90)

pd.crosstab(yr.Year, yr.Medal).plot(kind='bar', figsize=(18,8))
plt.title('Year wise Distribution for Medals',fontsize=15)
plt.xlabel('Years',fontsize=15)
plt.ylabel('No. of Medals',fontsize=15)
plt.show()
# plt.savefig('purchase_fre_job')


# ## 4. Season Wise Distribution

# In[59]:


# Participants in Winter and Summer Season
Sum = df[df.Season=='Summer']
Win = df[df.Season=='Winter']

sns.set(style="darkgrid")
plt.figure(figsize =(20,12))

plt.subplot(2,1,1)
sns.countplot(x= 'Year', data=Sum, palette= "Spectral")
plt.title('Participants In Summer Season', fontsize=15)

plt.subplot(2,1,2)
sns.countplot(x= 'Year', data=Win, palette= "Spectral")
plt.title('Participants In Winter Season', fontsize=15)


# In[60]:


# Compare all three medals in each year

# %matplotlib inline
# plt.figure(figsize=(20,30))
# plt.xticks(rotation=90)

pd.crosstab(df.Year, df.Season).plot(kind='bar', figsize=(18,8))
plt.title('Year wise Distribution for Medals',fontsize=15)
plt.xlabel('Years',fontsize=15)
plt.ylabel('No. of Participants',fontsize=15)
plt.show()
# plt.savefig('purchase_fre_job')


# In[61]:


# Medal in Winter and Summer Season
Sum = dff[dff.Season=='Summer']
Win = dff[dff.Season=='Winter']

sns.set(style="darkgrid")
plt.figure(figsize =(20,12))

plt.subplot(2,1,1)
sns.countplot(x= 'Year', data=Sum, palette= "Spectral")
plt.title('Winners of Medals In Summer Season', fontsize=15)

plt.subplot(2,1,2)
sns.countplot(x= 'Year', data=Win, palette= "Spectral")
plt.title('Winners of Medals In Winter Season', fontsize=15)


# In[62]:


# Medals in Winter and Summer season

pd.crosstab(dff.Year, dff.Season).plot(kind='bar', figsize=(18,8))
plt.title('Year wise Distribution for Medals',fontsize=15)
plt.xlabel('Years',fontsize=15)
plt.ylabel('No. of Medals',fontsize=15)
plt.show()


# In[63]:


womenOlympicsSum = dff[(dff.Gender=='F') & (dff.Season=='Summer')]
womenOlympicsWin = dff[(dff.Gender=='F') & (dff.Season=='Winter')]
# womenOlympicsSum.head()


# In[64]:


sns.set(style="darkgrid")
plt.figure(figsize =(20,12))

plt.subplot(2,1,1)
sns.countplot(x= 'Year', data=womenOlympicsSum, palette= "Spectral")
sns.lineplot(x='Year', data=womenOlympicsSum, palette='crest')
plt.title('Women Medallist In Summer Season', fontsize=15)

plt.subplot(2,1,2)
sns.countplot(x= 'Year', data=womenOlympicsWin, palette= "Spectral")
plt.title('Women Medallist In Winter Season', fontsize=15)


# In[65]:


women = dff[dff.Gender=='F']
pd.crosstab(women.Year, women.Season).plot(kind='bar', figsize=(18,8))
plt.title('Year wise Distribution for Medals',fontsize=15)
plt.xlabel('Years',fontsize=15)
plt.ylabel('No. of Medals',fontsize=15)
plt.show()


# #### Variation of Male/Female athletes over Time (Summer Games)

# In[66]:


MenOverTimeSum = dff[(dff.Gender == 'M') & (dff.Season == 'Summer')]
WomenOverTimeSum = dff[(dff.Gender == 'F') & (dff.Season == 'Summer')]
MenOverTimeWin = dff[(dff.Gender == 'M') & (dff.Season == 'Winter')]
WomenOverTimeWin = dff[(dff.Gender == 'F') & (dff.Season == 'Winter')]
womenOlympicsSum.head()


# In[67]:


part1 = MenOverTimeSum.groupby('Year')['Gender'].value_counts()
part2 = WomenOverTimeSum.groupby('Year')['Gender'].value_counts()
plt.figure(figsize=(20, 10))
part1.loc[:,'M'].plot(color='r')
part2.loc[:,'F'].plot(color='b')
plt.title('Variation of Male and Female Medallist over time for Summer Olympics', fontsize=15)


# In[68]:


part1 = MenOverTimeWin.groupby('Year')['Gender'].value_counts()
part2 = WomenOverTimeWin.groupby('Year')['Gender'].value_counts()
plt.figure(figsize=(20, 10))
part1.loc[:,'M'].plot(color='r')
part2.loc[:,'F'].plot(color='b')
plt.title('Variation of Male and Female Medallist over time for Winter Olympics', fontsize=15)


# In[69]:


# part2 = WomenOverTimeSum.groupby('Year')['Gender'].value_counts()
# plt.figure(figsize=(20, 10))
# part2.loc[:,'F'].plot(color='b')
# plt.title('Variation of Female Athletes over time', fontsize=15)


# #### Variation of Male/Female athletes over Time (Winter Games)

# In[70]:


MenOverTimeWin = dff[(dff.Gender == 'M') & (dff.Season == 'Winter')]
WomenOverTimeWin = dff[(dff.Gender == 'F') & (dff.Season == 'Winter')]


# In[71]:


# part1 = MenOverTimeWin.groupby('Year')['Gender'].value_counts()
# plt.figure(figsize=(20, 10))
# part1.loc[:,'M'].plot(color='r')
# plt.title('Variation of Male Athletes over time', fontsize=15)


# In[72]:


# part2 = WomenOverTimeWin.groupby('Year')['Gender'].value_counts()
# plt.figure(figsize=(20, 10))
# part2.loc[:,'F'].plot(color='b')
# plt.title('Variation of Female Athletes over time', fontsize=15)


# ## 5 Age & Height Wise Distribution

# In[73]:


# Medallist on the basis of Height and Weight
plt.figure(figsize=(20, 10))
ax = sns.scatterplot(x="Height", y="Weight", data=dff, hue="Medal", style="Medal")
plt.title('Height vs Weight of Olympic Medalists', fontsize=25)


# In[74]:


# Age distribution of the Athletes
plt.figure(figsize=(15,8))
plt.title("Age Distribution of the Athletes")
plt.xlabel("Age")
plt.ylabel("No. of Participants")
plt.hist(df.Age, bins = np.arange(10,80,1), color='orange', edgecolor = "white")
plt.show()


# In[75]:


# Age distribution of the Medalist
plt.figure(figsize=(15,8))
plt.title("Age Distribution of the Medalist")
plt.xlabel("Age")
plt.ylabel("No. of Medalist")
plt.hist(dff.Age, bins = np.arange(10,80,1), color='orange', edgecolor = "white")
plt.show()


# In[76]:


age_ab_60 = dff[dff.Age>=60]
age_ab_60


# In[77]:


age_ab_60[age_ab_60.Age>=60].Age.count()


# In[78]:


age_ab_60[['Name','Age','Medal','Sport','Region']]


# In[79]:


age_un_20 = dff[dff.Age<=20]
age_un_20


# ## 6. Sports Wise Distribution

# In[80]:


# Sports participated in Olympics
s = df.Sport.value_counts().sort_values(ascending=False)
plt.figure(figsize=(20,12))
plt.xticks(rotation=0)

sns.barplot(s, s.index)
plt.show()


# In[81]:


# Medals won by Sports
sp = dff.Sport.value_counts().sort_values(ascending=False)
sp


# In[82]:


# Find most
plt.figure(figsize=(20,12))
plt.xticks(rotation=0)

sns.barplot(sp, sp.index)
plt.show()


# In[83]:


dff.head()


# In[84]:


sp_medal = dff[['Sport', 'Medal']]
sp_medal.Medal.value_counts()


# In[85]:


g_medal = sp_medal[sp_medal.Medal == 'Gold'].Sport.value_counts().sort_values(ascending=False).head(50)
s_medal = sp_medal[sp_medal.Medal == 'Silver'].Sport.value_counts().sort_values(ascending=False).head(50)
# s_medal
b_medal = sp_medal[sp_medal.Medal == 'Bronze'].Sport.value_counts().sort_values(ascending=False).head(50)
b_medal.head()


# In[86]:


plt.figure(figsize=(20,15))
plt.xticks(rotation=90)

plt.subplot(1,3,1)
plt.title('Won Gold By Top 50 Sports', fontsize=15)
#plt.xlabel('Sports', fontsize=15)
plt.ylabel('Gold Medals', fontsize=15)
sns.barplot(x=g_medal, y=g_medal.index, palette = 'rocket');
#sns.lineplot(x=g_medal, y=g_medal.index, palette = 'crest');

plt.subplot(1,3,2)
plt.title('Won Silver By Top 50 Sports', fontsize=15)
#plt.xlabel('Sports', fontsize=15)
plt.ylabel('Silver Medals', fontsize=15)
sns.barplot(x=s_medal, y=s_medal.index, palette = 'rocket');
#sns.lineplot(x=s_medal, y=s_medal.index, color='b', palette = 'crest');

plt.subplot(1,3,3)
plt.title('Won Bronz By Top 50 Sports', fontsize=15)
plt.xlabel('Sports', fontsize=15)
plt.ylabel('Bronze Medals', fontsize=15)
sns.barplot(x=b_medal, y=b_medal.index, palette = 'rocket');
#sns.lineplot(x=b_medal, y=b_medal.index, color='b', palette = 'crest');

plt.tight_layout()


# In[87]:


# women = dff[dff.Gender == "F"]
# women
pd.crosstab(df.Sport, df.Medal).plot(kind='bar', figsize=(18,8))
plt.title('Medals Distributed by Sports',fontsize=15)
plt.xlabel('Sports',fontsize=12)
plt.ylabel('No. of Medals',fontsize=12)
plt.show()


# ## 7 Participants

# In[88]:


dff.isnull().sum()
# dff.Name


# In[89]:


participants1 = dfa.Name.value_counts().head(50)

participants2 = dff.Name.value_counts().head(50)
#participants.count()


# In[90]:


plt.figure(figsize=(20,25))
plt.xticks(rotation=0)

plt.subplot(2,1,1)
sns.barplot(x = participants1, y = participants1.index, palette="icefire")

plt.subplot(2,1,2)
sns.barplot(x = participants2, y = participants2.index, palette="icefire")


# ## 8 Country Comparison

# ### India

# In[91]:


ind = dff[(dff.Region=='India') & (dff.Medal == 'Gold')]
ind


# In[92]:


print('Total Gold Medal Wond By Indan Players : ', ind[ind['Medal']=='Gold'].Medal.count())
print('Total Silver Medal Wond By Indan Players : ', ind[ind['Medal']=='Silver'].Medal.count())
print('Total Bronze Medal Wond By Indan Players : ', ind[ind['Medal']=='Bronze'].Medal.count(), '\n')
print('Total Medal Wond By Indan Players : ', ind.Medal.value_counts().sum())
print('Total Participants For Sports Hockey : ', ind[ind['Sport']=='Hockey'].Sport.value_counts().sum())


# In[93]:


ind[ind['Sport']!='Hockey'].head()


# ### United State Of America (USA)

# In[94]:


usa = dff[dff.Region=='USA']
usa


# In[95]:


print('Total Gold Medal Wond By USA Players : ', usa[usa['Medal']=='Gold'].Medal.count())
print('Total Silver Medal Wond By USA Players : ', usa[usa['Medal']=='Silver'].Medal.count())
print('Total Bronze Medal Wond By USA Players : ', usa[usa['Medal']=='Bronze'].Medal.count(), '\n')
print('Total Medal Wond By USA Players : ', usa.Medal.value_counts().sum())


# ### 8 GDP wise distribution

# In[96]:


#import GDP
dfg = pd.read_csv('GDP1.csv')
#dfg.Region.sort_values(ascending=True).unique()
dfg


# In[97]:


df.Year.sort_values(ascending=True).unique()


# In[98]:


df_new = df.loc[df.Year.isin([1992,1994,1996,1998,2000,2002,2004,2006,2008,2010,2012,2014,2016])]
df_new.isnull().sum()


# In[99]:


df_new = df_new.merge(dfg, on=['Region','Year'], how='left')
df_new.isnull().sum()


# In[100]:


#df_new[df_new.isnull().any(axis=1)]
df_new[df_new.GDP.isnull()]


# In[101]:


#dfg.Region.sort_values(ascending=True).unique()


# In[102]:


#df_new.Region.sort_values(ascending=True).unique()


# In[103]:


df_new = df_new[df_new['GDP'].notna()]


# In[104]:


df_new.isnull().sum()


# In[105]:


df_new.shape


# In[106]:


df_new.isnull().sum()
dfm = df_new.dropna()
dfm.isnull().sum()


# In[107]:


dfm.reset_index(drop=True,inplace=True)
dfm


# In[108]:


# dfm.to_csv("dfg.csv")


# In[109]:


X = dfm[(dfm["Year"]==2016) & dfm["Region"]].sort_values(by='GDP',ascending=False)
# X


# In[110]:


plt.figure(figsize=(15,15))
plt.xticks(rotation=0)


a = X.Region
b = X.GDP
c = X.Region.value_counts()

plt.subplot(1,2,1)
plt.title('Country-GDP Distribution',fontsize=15)
plt.xlabel('Countries', fontsize=15)
plt.ylabel("GDP", fontsize=15)
sns.barplot(X.GDP, X.Region, palette = 'rocket');



plt.subplot(1,2,2)
plt.title('Country won Medals Distribution',fontsize=15)
plt.xlabel('No. of Medals', fontsize=15)
#plt.ylabel("Region", fontsize=15)
sns.barplot(c, c.index, palette = 'rocket');

plt.tight_layout()


# In[ ]:





# In[111]:


X1 = dfm[(dfm["Year"]==2014) & dfm["Region"]].sort_values(by='GDP',ascending=False)
X1.Region.unique()


# In[112]:


plt.figure(figsize=(15,15))
plt.xticks(rotation=0)


a = X1.Region
b = X1.GDP
c = X1.Region.value_counts()

plt.subplot(1,2,1)
plt.title('Country-GDP Distribution',fontsize=15)
plt.xlabel('Countries', fontsize=15)
plt.ylabel("GDP", fontsize=15)
sns.barplot(X1.GDP, X1.Region, palette = 'rocket');



plt.subplot(1,2,2)
plt.title('Country won Medals Distribution',fontsize=15)
plt.xlabel('No. of Medals', fontsize=15)
#plt.ylabel("Region", fontsize=15)
sns.barplot(c, c.index, palette = 'rocket');

plt.tight_layout()


# In[113]:


dfm[(dfm["Year"]==1996) & (dfm["Region"]=="USA")]


# In[114]:


X2 = dfm[(dfm["Year"]==2012) & dfm["Region"]].sort_values(by='GDP',ascending=False)
plt.figure(figsize=(15,15))
plt.xticks(rotation=0)


a = X2.Region
b = X2.GDP
c = X2.Region.value_counts()

plt.subplot(1,2,1)
plt.title('Country-GDP Distribution',fontsize=15)
plt.xlabel('Countries', fontsize=15)
plt.ylabel("GDP", fontsize=15)
sns.barplot(X2.GDP, X2.Region, palette = 'rocket');



plt.subplot(1,2,2)
plt.title('Country won Medals Distribution',fontsize=15)
plt.xlabel('No. of Medals', fontsize=15)
#plt.ylabel("Region", fontsize=15)
sns.barplot(c, c.index, palette = 'rocket');

plt.tight_layout()


# In[115]:


X3 = dfm[(dfm["Year"]==1992) & dfm["Region"]].sort_values(by='GDP',ascending=False)
plt.figure(figsize=(15,15))
plt.xticks(rotation=0)


a = X3.Region
b = X3.GDP
c = X3.Region.value_counts()

plt.subplot(1,2,1)
plt.title('Country-GDP Distribution',fontsize=15)
plt.xlabel('Countries', fontsize=15)
plt.ylabel("GDP", fontsize=15)
sns.barplot(X3.GDP, X3.Region, palette = 'rocket');



plt.subplot(1,2,2)
plt.title('Country won Medals Distribution',fontsize=15)
plt.xlabel('No. of Medals', fontsize=15)
#plt.ylabel("Region", fontsize=15)
sns.barplot(c, c.index, palette = 'rocket');

plt.tight_layout()


# ## Data Preprocessing

# In[116]:


df_new.columns


# In[117]:


# Select Required columns
dfn = df[['Gender', 'Age', 'Height', 'Weight', 'Season', 'Region','Sport','Event','Medal']]
dfn


# In[118]:


dfn.isnull().sum()


# In[119]:


le = LabelEncoder()
dfn['Gender']= le.fit_transform(dfn['Gender'])

dfn['Gender'].unique()


# In[120]:


print('Male =', 1, '\nFemale =', 0)


# In[121]:


dfn.Season.unique()


# In[122]:


dfn['Season']= le.fit_transform(dfn['Season'])


# In[123]:


print('\n 0 = Summer \n 1 = Winter')


# In[124]:


dfn.Medal.fillna('No Medal', inplace=True)
#dfn


# In[125]:


dfn.Medal.value_counts()


# In[126]:


dfn['Medal'].replace({'Gold' : 1, 'Silver' :  1, 'Bronze' : 1, 'No Medal' : 0}, inplace = True)


# In[127]:


dfn.Medal.value_counts()


# In[128]:


# dfn['Sport']= le.fit_transform(dfn['Sport'])
# dfn['Region'] = le.fit_transform(dfn['Region'])
dfn = pd.get_dummies(dfn, columns = ['Sport', 'Region', 'Event'], drop_first=True)

# dfn['Sport']= le.fit_transform(dfn['Sport'])
# dfn['Region']= le.fit_transform(dfn['Region'])
# dfn['Event']= le.fit_transform(dfn['Event'])


# In[129]:


dfn


# In[130]:


#dfn.Region.sort_values(ascending=True).unique()


# In[131]:


# dfn.Sport.sort_values(ascending=True).unique()


# ### Remove Outliers

# In[132]:


Q1 = dfn.Height.quantile(0.25)
Q3 = dfn.Height.quantile(0.75)
Q1, Q3


# In[133]:


IQR = Q3 - Q1
IQR


# In[134]:


lower_limit = Q1-1.5*IQR
upper_limit = Q3+1.5*IQR
lower_limit, upper_limit


# In[135]:


dfn = dfn[(dfn.Height>lower_limit) & (dfn.Height<upper_limit)]
dfn


# In[136]:


dfn.shape


# In[137]:


q1 = dfn.Age.quantile(0.25)
q3 = dfn.Age.quantile(0.75)
q1,q3


# In[138]:


IQR = q3-q1
IQR


# In[139]:


lower_limit = q1-1.5*IQR
upper_limit = q3+1.5*IQR
lower_limit, upper_limit


# In[140]:


dfn = dfn[(dfn.Age>13) & (dfn.Age<37)]
dfn


# In[141]:


Q1 = dfn.Weight.quantile(0.25)
Q3 = dfn.Weight.quantile(0.75)
Q1, Q3


# In[142]:


IQR = Q3 - Q1
IQR


# In[143]:


lower_limit = Q1-1.5*IQR
upper_limit = Q3+1.5*IQR
lower_limit, upper_limit


# In[144]:


dfn = dfn[(dfn.Weight>lower_limit) & (dfn.Weight<upper_limit)]
dfn


# In[145]:


dfn.reset_index(drop=True, inplace=True)
dfn


# In[146]:


#dfn.isnull().sum().sum()
dfn.head(50)


# In[ ]:





# In[ ]:





# In[147]:


# dfn.to_csv("Olympics_cd.csv")


# In[148]:


dfn.Medal.value_counts()


# In[149]:


# Dependent column
y = dfn['Medal']


# In[150]:


# independent columns
x = dfn.drop(columns=["Medal"], axis=1)


# In[151]:


y.value_counts()


# ## Data Standardization

# In[152]:


sc = StandardScaler()

x = sc.fit_transform(x)


# In[153]:


x.shape


# In[154]:


pca = PCA(n_components=2)
x = pca.fit_transform(x)
print(pca.explained_variance_ratio_)

#Explained variance
pca = PCA().fit(x)
plt.figure(figsize=(12,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# ## SMOTETomek

# In[155]:


# from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTETomek


# In[156]:


os=SMOTETomek(0.5)
x, y = os.fit_resample(x, y)


# ## Train Test Splitting

# In[157]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print('Total no. of samples: Training and Testing dataset separately!')
print('x_train:', np.shape(x_train))
print('y_train:', np.shape(y_train))
print('x_test:', np.shape(x_test))
print('y_test:', np.shape(y_test))


# ## Logistic Regression

# In[158]:


clf_Lr = LogisticRegression()
clf_Lr.fit(x_train,y_train)


# In[159]:


threshold = 0.34
lr_pred = (clf_Lr.predict_proba(x_test)[:, 1] > threshold).astype('float')
print(confusion_matrix(y_test,lr_pred), '\n')
print(accuracy_score(y_test,lr_pred), '\n')
print(classification_report(y_test,lr_pred))


# ## Random Forest Classifier

# In[160]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
# rf_pred = rf_model.predict(x_test)


# In[161]:


threshold = 0.4
rf_pred = (rf_model.predict_proba(x_test)[:, 1] > threshold).astype('float')
print(confusion_matrix(y_test,rf_pred), '\n')
print(accuracy_score(y_test,rf_pred), '\n')
print(classification_report(y_test,rf_pred))


# ## K Nearest Neighbor

# In[162]:


knn_model = KNeighborsClassifier().fit(x_train,y_train)
# knn_pred = knn_model.predict(x_test)


# In[163]:


threshold = 0.3
knn_pred = (knn_model.predict_proba(x_test)[:, 1] > threshold).astype('float')
print(confusion_matrix(y_test,knn_pred), '\n')
print(accuracy_score(y_test,knn_pred), '\n')
print(classification_report(y_test,knn_pred))


# ## Decision Tree

# In[164]:


from sklearn.tree import DecisionTreeClassifier 
dtc_model = DecisionTreeClassifier().fit(x_train,y_train)
# dtc_pred = dtc_model.predict(x_test)


# In[165]:


threshold = 0.35
dtc_pred = (dtc_model.predict_proba(x_test)[:, 1] > threshold).astype('float')
print(confusion_matrix(y_test,dtc_pred), '\n')
print(accuracy_score(y_test,dtc_pred), '\n')
print(classification_report(y_test,dtc_pred))


# ## Gradient Boosting Classifier

# In[166]:


from sklearn.ensemble import GradientBoostingClassifier
gbc_model = GradientBoostingClassifier().fit(x_train,y_train)
# gbc_pred = gbc_model.predict(x_test)


# In[167]:


threshold = 0.34
gbc_pred = (gbc_model.predict_proba(x_test)[:, 1] > threshold).astype('float')
print(confusion_matrix(y_test,gbc_pred), '\n')
print(accuracy_score(y_test,gbc_pred), '\n')
print(classification_report(y_test,gbc_pred))


# ## XGBoost Classifier

# In[168]:


from xgboost import XGBClassifier
xgb_model = XGBClassifier().fit(x_train,y_train)


# In[169]:


xgb_pred = xgb_model.predict(x_test)
print(confusion_matrix(y_test,xgb_pred), '\n')
print(accuracy_score(y_test,xgb_pred), '\n')
print(classification_report(y_test,xgb_pred))


# In[170]:


threshold = 0.34
xgb_pred = (xgb_model.predict_proba(x_test)[:, 1] > threshold).astype('float')
print(confusion_matrix(y_test,xgb_pred), '\n')
print(accuracy_score(y_test,xgb_pred), '\n')
print(classification_report(y_test,xgb_pred))


# In[172]:


print("Logistic Regression\n")
print(confusion_matrix(y_test,lr_pred),'\n')
print(accuracy_score(y_test,lr_pred), '\n')
print(classification_report(y_test,lr_pred))
print("---------------------------------------------------------------")

print("Random Forest Classifier\n")
print(confusion_matrix(y_test,rf_pred),'\n')
print(accuracy_score(y_test,rf_pred), '\n')
print(classification_report(y_test,rf_pred))
print("---------------------------------------------------------------")

print("KNearest Neighbor\n")
print(confusion_matrix(y_test,knn_pred))
print(accuracy_score(y_test,knn_pred), '\n')
print(classification_report(y_test,knn_pred))

print("---------------------------------------------------------------")
print("Decision Tree Classifier\n")
print(confusion_matrix(y_test,dtc_pred))
print(accuracy_score(y_test,dtc_pred), '\n')
print(classification_report(y_test,dtc_pred))
print("---------------------------------------------------------------")
print("Gradient Boosting Classifier\n")
print(confusion_matrix(y_test,gbc_pred))
print(accuracy_score(y_test,gbc_pred), '\n')
print(classification_report(y_test,gbc_pred))
print("---------------------------------------------------------------")
print("XGBoost\n")
print(confusion_matrix(y_test,xgb_pred))
print(accuracy_score(y_test,xgb_pred), '\n')
print(classification_report(y_test,xgb_pred))
print("---------------------------------------------------------------")


# In[ ]:




