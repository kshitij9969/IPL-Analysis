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
os.chdir(r"C:\Users\ks20092693\IPL_Analysis_Practice")

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
# Visualisation of Umpires
total_umpires.columns=['Umpire Name', 'Matches Umpired']
plt.xticks(rotation=90, fontsize=6)
sns.barplot(x='Umpire Name', y='Matches Umpired', data=total_umpires)

# Man of the matches
man_of_matches=matches['player_of_match'].value_counts().reset_index()
man_of_matches.columns=['Player Name', 'player_of_match']
# Visualisation of Man of the match
fig = plt.gcf()
fig.set_size_inches(18.5,10.5)
plt.xticks(rotation=45, fontsize=6)
plt.ylim(0,18)
sns.barplot(x='Player Name', y='player_of_match', data=man_of_matches.head(20))


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
