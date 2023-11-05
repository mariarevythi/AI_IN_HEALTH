#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Read csv file into a pandas dataframe
df = pd.read_csv(r'C:\Users\maria\Documents\MARIA\BioMed\FIRST_SEMESTER\ΑΙ\clinical_dataset.csv',delimiter=';')
# Take a look at the first few rows
df.head(10)
#print(df)


# In[2]:


# information for data 
from sklearn import preprocessing
df.info()


# In[3]:


# Convert nominal features to numerical
df['fried'].replace(['Frail', 'Pre-frail','Non frail'],
                        [2, 1,0], inplace=True)
df['gender'].replace(['F','M'],
                        [1,0], inplace=True)
df['ortho_hypotension'].replace(['Yes','No'],
                        [1,0], inplace=True)
df['vision'].replace(['Sees well','Sees moderately','Sees poorly'],
                        [2,1,0], inplace=True)
df['audition'].replace(['Hears well','Hears moderately','Hears poorly'],
                        [2,1,0], inplace=True)
df['weight_loss'].replace(['Yes','No'],
                        [1,0], inplace=True)
df['balance_single'].replace(['>5 sec','<5 sec','test non realizable'],
                        [2,1,0], inplace=True)
df['gait_speed_slower'].replace(['Yes','No'],
                        [1,0], inplace=True)
df['grip_strength_abnormal'].replace(['Yes','No'],
                        [1,0], inplace=True)
df['low_physical_activity'].replace(['Yes','No'],
                        [1,0], inplace=True)
df['memory_complain'].replace(['Yes','No'],
                        [1,0], inplace=True)
df['sleep'].replace(['Permanent sleep problem','Occasional sleep problem','No sleep problem'],
                        [2,1,0], inplace=True)
df['living_alone'].replace(['Yes','No'],
                        [1,0], inplace=True)
df['leisure_club'].replace(['Yes','No'],
                        [1,0], inplace=True)
df['house_suitable_participant'].replace(['Yes','No'],
                        [1,0], inplace=True)
df['house_suitable_professional'].replace(['Yes','No'],
                        [1,0], inplace=True)
df['health_rate'].replace(['5-Excellent','4-Good','3 - Medium','2 - Bad','1 - Very bad'],
                        [5,4,3,2,1], inplace=True)
df['health_rate_comparison'].replace(['5 - A lot better','4 - A little better','3 - About the same','2 - A little worse','1 - A lot worse'],
                        [5,4,3,2,1], inplace=True)
df['activity_regular'].replace(['> 5 h per week','> 2 h and < 5 h per week','< 2 h per week','No'],
                        [3,2,1,0], inplace=True)
df['smoking'].replace(['Current smoker','Past smoker (stopped at least 6 months)','Never smoked'],
                        [2,1,0], inplace=True)

#print(df)
df.head()


# In[4]:


# test collumn
print (df['raise_chair_time'])
print (df['raise_chair_time'].isnull())


# In[6]:


print (df.isnull().sum())


# In[7]:


print(df.columns) 


# In[4]:


# Remove erroneous values
for i in df.columns:
    for j in range (len(df[i])):
        if df[i][j]==999 or type(df[i][j]).__name__=='str':
            df.at[j,i]=None
        # df[i][j]=None
print (df['raise_chair_time'])     


# In[5]:


# Handle missing values
for i in df.columns:
    mean=df[i].mean(skipna=True)
    for j in range (len(df[i])):
        if pd.isna(df[i][j]):df.at[j,i]=mean
print (df['raise_chair_time'])  




#for i in df.columns:
    #mean=df[i].mean(skipna=True)
    #df[i].replace(['NaN'],
                      #  [mean], inplace=True)


# In[38]:


df.gait_speed_slower.head(20)


# In[6]:


def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior


# In[7]:


def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)
    return p_x_given_y


# In[8]:


def naive_bayes_categorical(df, X, Y):
    # get feature names
    features = list(df.columns)
    features.remove(Y)
    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 


# In[9]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=.3, random_state=41)
new_train=df.loc[:,df.columns!='weight_loss']
new_train=new_train.loc[:,new_train.columns!='exhaustion_score']
new_train=new_train.loc[:,new_train.columns!='gait_speed_slower']
new_train=new_train.loc[:,new_train.columns!='grip_strength_abnormal']
new_train=new_train.loc[:,new_train.columns!='low_physical_activity']
ind=[]
for i in range(len(train.columns)):
    ind.append(i)
ind.remove(1) 
ind.remove(9)
ind.remove(10)
ind.remove(16)
ind.remove(17)
ind.remove(18)
X_test = test.iloc[:,ind].values
Y_test = test.iloc[:,1].values
Y_pred = naive_bayes_categorical(new_train, X=X_test, Y="fried")
labels=[0,1,2]
from sklearn.metrics import confusion_matrix, f1_score
print(confusion_matrix(Y_test, Y_pred,labels=labels))
print(f1_score(Y_test, Y_pred,average='weighted'))


# In[10]:


print(Y_pred)


# In[34]:


print(Y_test)


# In[10]:


df.head(10)


# In[11]:


clinical_df=df


# In[12]:





# In[12]:


# Importing libraries
import pandas as pd
# Read csv file into a pandas dataframe
df = pd.read_csv(r'C:\Users\maria\Documents\MARIA\BioMed\FIRST_SEMESTER\ΑΙ\beacons_dataset.csv',delimiter=';')
# Take a look at the first few rows
df.head(10)


# In[13]:


# different categories for room column
df['room'].value_counts().index
# rename categories
df= df.replace(dict.fromkeys(['Bedroom1','Bedroom2','Chambre','Bedroom1st','Bedroom-1','2ndRoom','Bed','bedroom'], 'Bedroom'))
df= df.replace(dict.fromkeys(['Kitcen','Kitvhen', 'Kitchen2', 'Kitcheb','Kiychen', 'Kitch', 'Kithen','kitchen','Kichen'], 'Kitchen'))
df= df.replace(dict.fromkeys(['Leavingroom','Sittingroom','SittingRoom','LivingRoom', 'Livingroom1', 'Livingroom2', 'SittingOver','LuvingRoom', 'SeatingRoom','Living','LivibgRoom','Livroom','livingroom','Livingroon','Leavivinroom','TV','Sittigroom',
       'Luvingroom1','LivingRoom2', 'Sittinroom','Liningroom', 'Sitingroom', 'LeavingRoom'], 'Livingroom'))
df= df.replace(dict.fromkeys(['DinerRoom','Dinerroom','DiningRoom','DinningRoom'], 'DinnerRoom'))
df= df.replace(dict.fromkeys(['Desk','Office-2','Office1','Workroom','Office1st','Office2'], 'Office'))
df= df.replace(dict.fromkeys(['ExitHall'], 'Hall'))
df= df.replace(dict.fromkeys(['Entry'], 'Entrance'))
df= df.replace(dict.fromkeys(['Garage','Pantry'], 'Storage'))
df= df.replace(dict.fromkeys(['Bathroom1','Baghroom','Bathroom-1','Washroom','Bqthroom','Bathroim', 'Bsthroom','Bathroon','Barhroom',], 'Bathroom'))
df= df.replace(dict.fromkeys(['Veranda','Guard','Garden'], 'Outdoor'))
df= df.replace(dict.fromkeys(['Box', 'One','Two','Box-1','Four','Right','Three','three','Left', 'K','T' ], 'Other'))
df= df.replace(dict.fromkeys(['LaundryRoom'], 'Laundry'))


# In[14]:


# Define a function to check if a string is a 4-digit number
def is_4_digit(s):
    try:
        int(s)
        return len(s) == 4
    except ValueError:
        return False

# Use the function to filter the 'part_id' column
df = df[df['part_id'].apply(is_4_digit)]

# Print the resulting DataFrame
print(df)


# In[15]:


# filter dataframe for specific 4 places
df = df[df.room.isin(['Bedroom', 'Bathroom', 'Livingroom', 'Kitchen'])]

#group by patient and place
df_time_spent = df.groupby(['part_id', 'room']).size()

#create a new dataframe with total time spent by each patient
df_total_time = df_time_spent.groupby(level=0).sum()

# Divide the time spent in each room by the total time spent to get the percentage
df_percentage = df_time_spent.div(df_total_time, level=0)

# Print the resulting DataFrame
print(df_percentage)


# In[16]:


df['part_id'] = pd.to_numeric(df['part_id'], downcast='integer')


# In[17]:


# Use the `merge()` function to combine the dataframes on the 'part_id' column
merged_df = pd.merge(clinical_df, df, on='part_id')

# Print the resulting DataFrame
print(merged_df)


# In[ ]:




