# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:04:07 2019

@author: Kshitij Singh
Description: Contains different analysis of IPL 2008-2017
"""
# 1. Importing the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import os
import seaborn as sns

py.init_notebook_mode(connected=True)
plt.style.use('fivethirtyeight')
os.chdir(r"/Users/kshitijsingh/Downloads/IPL-Analysis") # For macOS
os.chdir(r"C:\Users\ks20092693\IPL_Analysis_Practice") # For windows

# 2. Importing the datasets
matches = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

# 3. Cleaning the dataset
# Drop the upmire 3 column as it is mostly NaN
matches.drop(['umpire3'],
             axis=1,
             inplace=True)
# Replacing all NaN with 0
delivery.fillna(0, inplace=True)

# Replacing team names with abrevations
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore',
                 'Deccan Chargers','Chennai Super Kings','Rajasthan Royals','Delhi Daredevils',
                 'Gujarat Lions','Kings XI Punjab','Sunrisers Hyderabad','Rising Pune Supergiants',
                 'Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant'],
                ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW',
                 'RPS'],
                 inplace=True)
delivery.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore',
                 'Deccan Chargers','Chennai Super Kings','Rajasthan Royals','Delhi Daredevils',
                 'Gujarat Lions','Kings XI Punjab','Sunrisers Hyderabad','Rising Pune Supergiants',
                 'Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant'],
                ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW',
                 'RPS'],
                 inplace=True)

# 4. Calculating basic parameters

# Storing the team names
teams_names = matches['team1'].unique() 
 # Total_matches played since 2008 in IPL
total_matches = matches.shape[0]
 # Cities where matches were played 
cities = matches['city'].unique() 
# All venues where matches were played
total_venues = matches['city'].nunique() 
# List of all umpires 
total_umpires = pd.concat([matches['umpire1'],matches['umpire2']])
total_umpires = total_umpires.value_counts().reset_index()
# Man of the matches
man_of_matches=matches['player_of_match'].value_counts().reset_index()
man_of_matches.columns=['Player Name', 'player_of_match']

# Visualisation of Umpires
total_umpires.columns=['Umpire Name', 'Matches Umpired']
plt.xticks(rotation=90, fontsize=6)
sns.barplot(x='Umpire Name', y='Matches Umpired', data=total_umpires)
# Visualisation of Man of the match
fig = plt.gcf()
fig.set_size_inches(18.5,10.5)
plt.xticks(rotation=45, fontsize=6)
plt.ylim(0,18)
sns.barplot(x='Player Name', y='player_of_match', data=man_of_matches.head(20))


# Computing and storing batsman data


# Calculating batsman parameters and storing them in batting
batting = delivery.groupby(['batsman'])['ball'].count().reset_index()

runs = delivery.groupby(['batsman'])['batsman_runs'].sum().reset_index() # Total runs
# Ones
ones = delivery.groupby(['batsman'])['batsman_runs'].agg(lambda x: (x==1).sum()).reset_index() 
ones = ones.rename(columns={'batsman_runs':'ones'})
# Twos
twos = delivery.groupby(['batsman'])['batsman_runs'].agg(lambda x: (x==2).sum()).reset_index() 
twos = twos.rename(columns={'batsman_runs':'twos'})
# Threes
threes = delivery.groupby(['batsman'])['batsman_runs'].agg(lambda x: (x==3).sum()).reset_index() 
threes = threes.rename(columns={'batsman_runs':'threes'})
# Fours
fours = delivery.groupby(['batsman'])['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index() 
fours = fours.rename(columns={'batsman_runs':'fours'})
# Fives
fives = delivery.groupby(['batsman'])['batsman_runs'].agg(lambda x: (x==5).sum()).reset_index() 
fives = fives.rename(columns={'batsman_runs':'fives'})
# Sixes
sixes = delivery.groupby(['batsman'])['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index() 
sixes = sixes.rename(columns={'batsman_runs':'sixes'})
# Highest score
highest=delivery.groupby(["match_id", "batsman","batting_team"])['batsman_runs'].sum().reset_index()
highest = highest.groupby(["batsman","batting_team"])["batsman_runs"].max().reset_index()
highest = highest.groupby(["batsman"])["batsman_runs"].max().reset_index()
highest = highest.rename(columns={'batsman_runs':'highest'})
# Matches played
matches_played = delivery.groupby(['batsman'])['match_id'].nunique().reset_index()
matches_played=matches_played.rename(columns={'match_id':'matches'})
# Man of the match
man_of_match= matches['player_of_match'].value_counts().reset_index()
man_of_match['batsman']=man_of_match['index']
man_of_match.drop(['index'], axis = 1, inplace=True)
# Score by a player in matches
scores_by_batsman= delivery.groupby(['match_id','batsman'])['batsman_runs'].sum().reset_index()
# Hundreds
hundreds = scores_by_batsman.groupby(['batsman'])['batsman_runs'].agg(lambda x: (x>=100).sum()).reset_index()
hundreds = hundreds.rename(columns={'batsman_runs':'hundreds'})
# Fifties
fifties = scores_by_batsman.groupby(['batsman'])['batsman_runs'].agg(lambda x: np.logical_and(x>=50, x<100).sum()).reset_index()
fifties = fifties.rename(columns={'batsman_runs':'fifties'})
# Total runs
total_scores = delivery.groupby(['batsman'])['batsman_runs'].agg(lambda x: x.sum()).reset_index()
# Teams played for
teams_played = delivery.groupby(['batsman'])['batting_team'].nunique()
teams_played = pd.DataFrame(teams_played)
teams_played['batsman']=teams_played.index
teams_played.index.name = None # To avoid ambigious index error
teams_played = teams_played.rename(columns={'batting_team':'teams_played'})
teams_played= teams_played[:-1]
# Run outs
run_out = delivery.groupby(['batsman'])['dismissal_kind'].agg(lambda x: (x=='run out').sum()).reset_index()
run_out = run_out.rename(columns={'dismissal_kind':'run_outs'})

# Merging with batting


# Merging runs and renaming appropriatly
batting = batting.merge(runs, left_on='batsman', right_on='batsman', how='outer')
batting = batting.rename(columns={'batsman':'batsman','ball':'balls_played','batsman_runs':'runs_scored'})

# Merging ones, twos, threes, fours, fives and sixes and renaming appropriatly
batting = batting.merge(ones, left_on='batsman', right_on='batsman', how='outer')
batting = batting.merge(twos, left_on='batsman', right_on='batsman', how='outer')
batting = batting.merge(threes, left_on='batsman', right_on='batsman', how='outer')
batting = batting.merge(fours, left_on='batsman', right_on='batsman', how='outer')
batting = batting.merge(fives, left_on='batsman', right_on='batsman', how='outer')
batting = batting.merge(sixes, left_on='batsman', right_on='batsman', how='outer')
batting = batting.merge(highest, left_on="batsman", right_on="batsman", how="outer")
batting = batting.merge(matches_played, left_on="batsman", right_on="batsman", how="outer")
batting = batting.merge(man_of_match, left_on='batsman', right_on='batsman', how='outer')
batting = batting.merge(hundreds, left_on='batsman', right_on='batsman', how='outer')
batting = batting.merge(fifties, left_on='batsman', right_on='batsman', how='outer')
batting = batting.merge(teams_played, left_on='batsman', right_on='batsman', how='outer')
batting = batting.merge(run_out, left_on='batsman', right_on='batsman', how='outer')


# Calculating strike rate
# Strike rate = runs scored per ball or strike rate = (runs scored)/(balls played)
batting['strike_rate'] = batting['runs_scored']*100/batting['balls_played']

# Calculating average runs
# Average runs = runs scored per match or average runs = (runs scored)/(matches played)
batting['average'] = batting['runs_scored']/batting['matches_played']


# Calculating bowler parameters

balls = delivery.groupby(['bowler'])['ball'].count().reset_index()
maiden = delivery.groupby(['bowler','over']) ## Not done
runs_conceded = delivery.groupby(['bowler'])['total_runs'].sum().reset_index()
runs_conceded = runs_conceded.rename(columns={'total_runs':'runs_conceded'})
fifer = delivery.groupby(['match_id','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
fifer = fifer.groupby(['bowler'])['player_dismissed'].agg(lambda x: (x>=5).sum()).reset_index()
fifer = fifer.rename(columns={'player_dismissed':'fifer'})
fours_w = delivery.groupby(['match_id','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
fours_w = fours_w.groupby(['bowler'])['player_dismissed'].agg(lambda x: (x==4).sum()).reset_index()
fours_w = fours_w.rename(columns={'player_dismissed':'fours'})

threes_w = delivery.groupby(['match_id','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
threes_w = threes_w.groupby(['bowler'])['player_dismissed'].agg(lambda x: (x==3).sum()).reset_index()
threes_w = threes_w.rename(columns={'player_dismissed':'threes'})

twos_w = delivery.groupby(['match_id','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
twos_w = twos_w.groupby(['bowler'])['player_dismissed'].agg(lambda x: (x==2).sum()).reset_index()
twos_w = twos_w.rename(columns={'player_dismissed':'twos'})

ones_w = delivery.groupby(['match_id','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
ones_w = ones_w.groupby(['bowler'])['player_dismissed'].agg(lambda x: (x==1).sum()).reset_index()
ones_w = ones_w.rename(columns={'player_dismissed':'ones'})

wides = delivery.groupby(['bowler'])['wide_runs'].agg(lambda x: x.sum()).reset_index()
bye_runs = delivery.groupby(['bowler'])['bye_runs'].agg(lambda x: x.sum()).reset_index()
no_balls_runs = delivery.groupby(['bowler'])['noball_runs'].agg(lambda x: x.sum()).reset_index()
penalty_runs = delivery.groupby(['bowler'])['penalty_runs'].agg(lambda x: x.sum()).reset_index()
extra_runs = delivery.groupby(['bowler'])['extra_runs'].agg(lambda x: x.sum()).reset_index()
caught = delivery.groupby(['bowler'])['dismissal_kind'].agg(lambda x: (x=='caught').sum()).reset_index()
caught =  caught.rename(columns={'dismissal_kind':'caught'})
bowled = delivery.groupby(['bowler'])['dismissal_kind'].agg(lambda x: (x=='bowled').sum()).reset_index()
bowled = bowled.rename(columns={'dismissal_kind':'bowled'})
caught_and_bowled = delivery.groupby(['bowler'])['dismissal_kind'].agg(lambda x: (x=='caught and bowled').sum()).reset_index()
caught_and_bowled = caught_and_bowled.rename(columns={'dismissal_kind':'caught_and_bowled'})
lbw =  delivery.groupby(['bowler'])['dismissal_kind'].agg(lambda x: (x=='lbw').sum()).reset_index()
lbw = lbw.rename(columns={'dismissal_kind':'lbw'})


# Merging with bowling

bowler = delivery.groupby(['bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
bowler = bowler.rename(columns={'player_dismissed':'wickets'})
bowler = bowler.merge(balls, left_on='bowler', right_on='bowler', how='outer')
# bowler = bowler.merge(maiden, left_on='bowler', right_on='bowler', how='outer') # Not done
bowler = bowler.merge(runs_conceded, left_on='bowler', right_on='bowler', how='outer')
bowler = bowler.merge(fifer, left_on='bowler', right_on='bowler', how='outer')
bowler = bowler.merge(wides, left_on='bowler', right_on='bowler', how='outer')
bowler = bowler.merge(bye_runs, left_on='bowler', right_on='bowler', how='outer')
bowler = bowler.merge(no_balls_runs, left_on='bowler', right_on='bowler', how='outer')
bowler = bowler.merge(penalty_runs, left_on='bowler', right_on='bowler', how='outer')
bowler = bowler.merge(extra_runs, left_on='bowler', right_on='bowler', how='outer')
bowler = bowler.merge(caught, left_on='bowler', right_on='bowler', how='outer')
bowler = bowler.merge(bowled, left_on='bowler', right_on='bowler', how='outer')
bowler = bowler.merge(caught_and_bowled, left_on='bowler', right_on='bowler', how='outer')
bowler = bowler.merge(lbw, left_on='bowler', right_on='bowler', how='outer')


# Calculating economy 
bowler['economy'] = bowler['runs_conceded']*6/(bowler['ball'])


############ Not done yet ############ Ducks
ducks = delivery.groupby(['match_id','batsman'])['batsman_runs']

ducks = delivery.groupby(['match_id','batsman'])['batsman_runs'].agg(lambda x: (x==0).sum()).reset_index()
ducks.drop(['match_id'], axis = 1, inplace=True)
balls = balls.merge(ducks, left_on='batsman', right_on='batsman', how='outer')

balls.drop(['batsman_runs'], axis=1, inplace=True)

#### Analysis of bowlers ####

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
temp_bowler = bowler[['bowler','economy','wickets']].copy()

temp_train = sc_X.fit_transform(temp_bowler[['economy','wickets']])

temp_train = pd.DataFrame(temp_train)
temp_train = temp_train.rename(columns={'1':'wickets'})
plt.scatter(temp_train['0'], temp_train['1'])



############ Not done yet ############


player_stats.insert(0,'player names',player_names, allow_duplicates=False)
player_stats.drop_duplicates(subset='player names',keep='first',inplace=True)
player_stats.reset_index()
player_stats.loc['player name']=pd.concat([delivery['batsman'],delivery['non_striker']])
player_stats = player_stats.drop('fours',axis=1)

player_stats.insert(1, 'fours',np.zeros(465), True)

temp = pd.DataFrame()
temp['striker'] = delivery['batsman']
temp['runs']= delivery['batsman_runs']

fours=delivery.groupby('batsman')['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index()
sixes=delivery.groupby('batsman')['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index()


df2 = temp.groupby(['striker'])[temp['runs']==4].count().reset_index()

df = delivery.groupby(['batsman'])['ball'].count().reset_index()
player_stats.set_index('player_name',inplace=True)
player_stats['balls'] = df['ball']
sns.countplot(x='player_name', data=matches, palette=sns.color_palette('winter'))
df1 = df.first()

print(df.first())




##### Perform K-means #####

np.random.seed(200)
plt.scatter()
temp1= balls.loc[balls['balls_played']>200]
temp = pd.DataFrame({
        'strike_rate':temp1['strike_rate'],
        'average':temp1['average']   
        })


plt.scatter(temp['average'], temp['strike_rate'])

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5)
labels=kmeans.fit(temp)
temp=temp.fillna(0)





labels = kmeans.predict(temp)
centroid = kmeans.cluster_centers_



centroids = kmeans.cluster_centers_
colmap = {1:'r',2:'g',3:'b', 4:'y', 5:'k'}
fig = plt.figure(figsize=(5,5))
colors=map(lambda x: colmap[x+1], labels)
color1=list(colors)
plt.scatter(temp['average'],temp['strike_rate'],color=color1,alpha=0.5,edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])

plt.xlim(0,50)
plt.ylim(0,250)
plt.show()



##### Perform K-means #####

















################# Practice #################

ax = matches['toss_winner'].value_counts().plot.bar(width=0.9, color=sns.color_palette('RdYlGn',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

plt.show()


matches_played_byteams=pd.concat([matches['team1'],matches['team2']])
matches_played_byteams=matches_played_byteams.value_counts().reset_index()
matches_played_byteams.columns=['Team','Total Matches']
matches_played_byteams.set_index('Team',inplace=True)
matches_played_byteams['wins']=matches['winner'].value_counts().reset_index()['winner']





highest_match_win=matches['winner'].value_counts().idxmax()


highest_match_win_by_run = matches.iloc[[matches['win_by_runs'].idxmax()]]
toss_decision = (matches['toss_decision'].value_counts())/577*100

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
sns.countplot(x='season', hue='toss_decision', data=matches)
plt.show()

highest_toss_winners = matches['toss_winner'].value_counts().plot.bar(width=0.9,color=sns.color_palette('RdYlGn',20))
for i in highest_toss_winners.patches:
    highest_toss_winners.annotate(format(i.get_height()),(i.get_x()+0.15,i.get_height()+1))
plt.show()

matches_played_by_teams = pd.concat([matches['team1'], matches['team2']])
matches_played_by_teams = matches_played_by_teams.value_counts().reset_index()
matches_played_by_teams.columns = ['Teams', 'Matches Won']
matches_played_by_teams['wins'] = matches['winner'].value_counts().reset_index()['winner'] 
# Doubt about reset_index(). Why did we add ['winner']?
matches_played_by_teams.set_index('Teams', inplace=True)

trace1 = go.Bar(
        x=matches_played_by_teams.index,
        y=matches_played_by_teams['Matches Won'],
        name = 'Total Matches')
trace2 = go.Bar(
        x=matches_played_by_teams.index,
        y=matches_played_by_teams['wins'],
        name = 'Total Wins')
import plotly
plotly.__version__
data = [trace1, trace2]

layout = go.Layout(
        barmode='stack')
fig = go.Figure(data=data, layout=layout)
temp = py.plot(fig, filename='stacked-bar1.html', auto_open=True)
fig.show()

fig = plt.gcf()
df = matches[matches['toss_winner']==matches['winner']]
slices =[len(df), (total_matches-len(df))]
labels = ['yes', 'no']
plt.pie(slices, labels=labels, startangle=90, shadow=True, explode=(0,0.05),autopct='%1.0f%%',colors=['r','g'])
