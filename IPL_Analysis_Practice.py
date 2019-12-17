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



# Teams and team players across seasons

teams = pd.DataFrame()
teams[['players','MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW',
                 'RPS']] = [np.nan,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

teams.insert(0, 'players', np.nan, allow_duplicates=False)
teams.insert(1, 'MI', 0, allow_duplicates=False)
teams.insert(2, 'KKR', 0, allow_duplicates=False)
teams.insert(3, 'RCB', 0, allow_duplicates=False)
teams.insert(4, 'DC', 0, allow_duplicates=False)
teams.insert(5, 'CSK', 0, allow_duplicates=False)
teams.insert(6, 'RR', 0, allow_duplicates=False)
teams.insert(7, 'DD', 0, allow_duplicates=False)
teams.insert(8, 'GL', 0, allow_duplicates=False)
teams.insert(9, 'KXIP', 0, allow_duplicates=False)
teams.insert(10, 'SRH', 0, allow_duplicates=False)
teams.insert(11, 'RPS', 0, allow_duplicates=False)
teams.insert(12, 'KTK', 0, allow_duplicates=False)
teams.insert(13, 'PW', 0, allow_duplicates=False)



teams['players']= batting['batsman']
players_played = delivery.groupby(['batting_team','batsman'])['match_id'].nunique()
temp_index = players_played.index
players_played = pd.DataFrame(players_played)
players_played.insert(1, 'temp_index', temp_index)
players_played.index.name = None
players_played = players_played[:-1]
players_played['team']= players_played['temp_index'].str[0]
players_played['batsman']= players_played['temp_index'].str[1]
players_played = players_played.set_index('batsman')
print(players_played.index)
players_played['batsman']= players_played.index
players_played = players_played.reset_index()
temp_players_played = delivery.groupby(['batting_team','batsman'])['match_id'].nunique()
players_played.drop('temp_index', axis=1, inplace=True)




players_played_bowler = delivery.groupby(['bowling_team','bowler'])['match_id'].nunique()
temp_index = players_played_bowler.index
players_played_bowler = pd.DataFrame(players_played_bowler)

players_played_bowler.insert(1, 'temp_index', temp_index)

players_played_bowler['team']= players_played_bowler['temp_index'].str[0]
players_played_bowler['bowler']= players_played_bowler['temp_index'].str[1]
players_played_bowler = players_played_bowler.set_index('bowler')
players_played_bowler['bowler']= players_played_bowler.index
players_played_bowler.drop(['temp_index'], axis=1, inplace=True)
players_played_bowler.index.name = None

MI_top_bowlers = pd.DataFrame(players_played_bowler[players_played_bowler['team']=='MI']).sort_values('match_id', ascending=False).head(10)
CSK_top_bowlers = pd.DataFrame(players_played_bowler[players_played_bowler['team']=='CSK']).sort_values('match_id', ascending=False).head(10) 
RCB_top_bowlers = pd.DataFrame(players_played_bowler[players_played_bowler['team']=='RCB']).sort_values('match_id', ascending=False).head(10) 
MI_top_bowlers.insert(3, 'economy',np.nan)
MI_top_bowlers.insert(4, 'wickets',np.nan)
MI_top_bowlers.drop(['economy','wickets'], axis=1, inplace=True)

CSK_top_bowlers.insert(3, 'economy',np.nan)
CSK_top_bowlers.insert(4, 'wickets',np.nan)
CSK_top_bowlers.drop(['economy','wickets'], axis=1, inplace=True)


for bowl in MI_top_bowlers['bowler']:
    print(bowl)
    MI_top_bowlers[MI_top_bowlers['bowler']==bowl]['economy']=bowler[bowler['bowler']==bowl]['economy']

MI_top_bowlers = MI_top_bowlers.merge(bowler, left_on='bowler', right_on='bowler', how='inner')
CSK_top_bowlers = CSK_top_bowlers.merge(bowler, left_on='bowler', right_on='bowler', how='inner')
RCB_top_bowlers = RCB_top_bowlers.merge(bowler, left_on='bowler', right_on='bowler', how='inner')


plt.scatter(MI_top_bowlers['economy'], MI_top_bowlers['wickets'], color='blue')
plt.scatter(CSK_top_bowlers['economy'], CSK_top_bowlers['wickets'], color='yellow')
plt.scatter(RCB_top_bowlers['economy'], RCB_top_bowlers['wickets'], color='red')

print(delivery.groupby(['batting_team']))



#### Segregating death over bowler, power play bowlers and middle over bowler

death_over_bowlers = pd.DataFrame(delivery.groupby(['bowling_team','over'])['bowler'])
death_over_bowlers.columns = death_over_bowlers.columns.astype(str)
temp = death_over_bowlers['0']
death_over_bowlers['team'] = temp.str[0]
death_over_bowlers['over'] = temp.str[1]
death_over_bowlers.drop(['0'], axis=1, inplace=True)
temp = death_over_bowlers['1']

temp1 = delivery[delivery['over']==20]
temp2 = temp1.groupby(['bowling_team'])
temp2 = delivery[['bowling_team','bowler']][delivery['over']==20]

temp_bowler_count = temp2.groupby(['bowling_team','bowler'])['bowler'].count()
temp_bowler_count = pd.DataFrame(temp_bowler_count)
temp_bowler_count['temp_1'] = temp_bowler_count.index
temp_bowler_count['team'] = temp_bowler_count['temp_1'].str[0]
temp_bowler_count['player'] = temp_bowler_count['temp_1'].str[1]
temp_bowler_count.drop(['temp_1'], axis=1, inplace=True)
temp_bowler_count.index.name = None
temp_bowler_count = temp_bowler_count.rename(columns={'bowler':'death_overs_bowled'})
temp_bowler_count = temp_bowler_count[:-2]

death_over_stats = delivery[delivery['over']==20]
death_over_stats = death_over_stats.groupby(['bowling_team','bowler'])['total_runs'].sum().reset_index()
death_over_balls = delivery[delivery['over']==20].groupby(['bowling_team','bowler'])['ball'].count().reset_index()
death_over_stats['economy'] = death_over_stats['total_runs']*6/death_over_stats['ball']
death_over_wickets = delivery[delivery['over']==20].groupby(['bowling_team','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()



death_over_stats = death_over_stats.merge(death_over_balls, left_on=['bowler','bowling_team'], right_on=['bowler', 'bowling_team'], how='outer')
death_over_stats = death_over_stats.merge(death_over_wickets, left_on=['bowler','bowling_team'], right_on=['bowler','bowling_team'], how='outer')
temp_bowler_count = temp_bowler_count.merge(death_over_stats['total_runs'], left_on='player', right_on='bowler', how='outer')

temp_bowler_count = temp_bowler_count.rename(columns={'team':'bowling_team','player':'bowler'}).reset_index()
temp_bowler_count.index.name = None
death_over_stats.index.name = None
death_over_stats = death_over_stats.merge(temp_bowler_count, left_on='bowler', right_on='bowler', how='outer')


#### Opening bowler ####
temp2 = delivery[['bowling_team','bowler']][delivery['over']==1]
temp_bowler_count = temp2.groupby(['bowling_team','bowler'])['bowler'].count()
temp_bowler_count = pd.DataFrame(temp_bowler_count)
temp_bowler_count['temp_1'] = temp_bowler_count.index
temp_bowler_count['team'] = temp_bowler_count['temp_1'].str[0]
temp_bowler_count['player'] = temp_bowler_count['temp_1'].str[1]
temp_bowler_count.drop(['temp_1'], axis=1, inplace=True)
temp_bowler_count.index.name = None
temp_bowler_count = temp_bowler_count.rename(columns={'first_over_bowled':'first_over_balls'})
temp_bowler_count = temp_bowler_count[:-2]

first_over_stats = delivery[delivery['over']==1]
first_over_stats = first_over_stats.groupby(['bowling_team','bowler'])['total_runs'].sum().reset_index()
first_over_balls = delivery[delivery['over']==1].groupby(['bowling_team','bowler'])['ball'].count().reset_index()
first_over_wickets = delivery[delivery['over']==1].groupby(['bowling_team','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()

first_over_stats = first_over_stats.merge(first_over_balls, left_on=['bowling_team','bowler'], right_on=['bowling_team','bowler'], how='outer')
first_over_stats = first_over_stats.merge(first_over_wickets, left_on=['bowling_team','bowler'], right_on=['bowling_team','bowler'], how='outer')

first_over_stats['economy']=first_over_stats['total_runs']*6/first_over_stats['ball']

#### 19th Over Stats ####
temp2 = delivery[['bowling_team','bowler']][delivery['over']==19]
temp_bowler_count = temp2.groupby(['bowling_team','bowler'])['bowler'].count()
temp_bowler_count = pd.DataFrame(temp_bowler_count)
temp_bowler_count['temp_1'] = temp_bowler_count.index
temp_bowler_count['team'] = temp_bowler_count['temp_1'].str[0]
temp_bowler_count['player'] = temp_bowler_count['temp_1'].str[1]
temp_bowler_count.drop(['temp_1'], axis=1, inplace=True)
temp_bowler_count.index.name = None
temp_bowler_count = temp_bowler_count.rename(columns={'bowler':'19_over_bowled'})
temp_bowler_count = temp_bowler_count[:-2]

nineteenth_over_stats = delivery[delivery['over']==19]
nineteenth_over_stats = nineteenth_over_stats.groupby(['bowling_team','bowler'])['total_runs'].sum().reset_index()
nineteenth_over_balls = delivery[delivery['over']==19].groupby(['bowling_team','bowler'])['ball'].count().reset_index()
nineteenth_over_wickets = delivery[delivery['over']==19].groupby(['bowling_team','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()

nineteenth_over_stats = nineteenth_over_stats.merge(nineteenth_over_balls, left_on=['bowling_team','bowler'], right_on=['bowling_team','bowler'], how='outer')
nineteenth_over_stats = nineteenth_over_stats.merge(nineteenth_over_wickets, left_on=['bowling_team','bowler'], right_on=['bowling_team','bowler'], how='outer')

nineteenth_over_stats['economy']=nineteenth_over_stats['total_runs']*6/nineteenth_over_stats['ball']

#### Power play stats ####
temp2 = delivery[['bowling_team','bowler']][np.logical_and(delivery['over']>=1, delivery['over']<=6)]
temp_bowler_count = temp2.groupby(['bowling_team','bowler'])['bowler'].count()
temp_bowler_count = pd.DataFrame(temp_bowler_count)
temp_bowler_count['temp_1'] = temp_bowler_count.index
temp_bowler_count['team'] = temp_bowler_count['temp_1'].str[0]
temp_bowler_count['player'] = temp_bowler_count['temp_1'].str[1]
temp_bowler_count.drop(['temp_1'], axis=1, inplace=True)
temp_bowler_count.index.name = None
temp_bowler_count = temp_bowler_count.rename(columns={'bowler':'power_play_bowled'})
temp_bowler_count = temp_bowler_count[:-2]


power_play_bowling = delivery[np.logical_and(delivery['over']>=1, delivery['over']<=6)]
power_play_bowling = power_play_bowling.groupby(['bowling_team','bowler'])['total_runs'].sum().reset_index()
power_play_bowling_balls = delivery[np.logical_and(delivery['over']>=1, delivery['over']<=6)].groupby(['bowling_team','bowler'])['ball'].count().reset_index()
power_play_bowling_wickets = delivery[np.logical_and(delivery['over']>=1, delivery['over']<=6)].groupby(['bowling_team','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()

power_play_bowling = power_play_bowling.merge(power_play_bowling_balls, left_on=['bowling_team','bowler'], right_on=['bowling_team','bowler'], how='outer')
power_play_bowling = power_play_bowling.merge(power_play_bowling_wickets, left_on=['bowling_team','bowler'], right_on=['bowling_team','bowler'], how='outer')

power_play_bowling['economy']=power_play_bowling['total_runs']*6/power_play_bowling['ball']


#### Middle overs between 6 - 12 ####

temp2 = delivery[['bowling_team','bowler']][np.logical_and(delivery['over']>=7, delivery['over']<=12)]
temp_bowler_count = temp2.groupby(['bowling_team','bowler'])['bowler'].count()
temp_bowler_count = pd.DataFrame(temp_bowler_count)
temp_bowler_count['temp_1'] = temp_bowler_count.index
temp_bowler_count['team'] = temp_bowler_count['temp_1'].str[0]
temp_bowler_count['player'] = temp_bowler_count['temp_1'].str[1]
temp_bowler_count.drop(['temp_1'], axis=1, inplace=True)
temp_bowler_count.index.name = None
temp_bowler_count = temp_bowler_count.rename(columns={'bowler':'power_play_bowled'})
temp_bowler_count = temp_bowler_count[:-2]


Middle_overs_six_twelve = delivery[np.logical_and(delivery['over']>=7, delivery['over']<=12)]
Middle_overs_six_twelve = Middle_overs_six_twelve.groupby(['bowling_team','bowler'])['total_runs'].sum().reset_index()
Middle_overs_six_twelve_balls = delivery[np.logical_and(delivery['over']>=7, delivery['over']<=12)].groupby(['bowling_team','bowler'])['ball'].count().reset_index()
Middle_overs_six_twelve_wickets = delivery[np.logical_and(delivery['over']>=7, delivery['over']<=12)].groupby(['bowling_team','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()

Middle_overs_six_twelve = Middle_overs_six_twelve.merge(Middle_overs_six_twelve_balls, left_on=['bowling_team','bowler'], right_on=['bowling_team','bowler'], how='outer')
Middle_overs_six_twelve = Middle_overs_six_twelve.merge(Middle_overs_six_twelve_wickets, left_on=['bowling_team','bowler'], right_on=['bowling_team','bowler'], how='outer')

Middle_overs_six_twelve['economy']=Middle_overs_six_twelve['total_runs']*6/Middle_overs_six_twelve['ball']


#### Middle overs 12 - 18
temp2 = delivery[['bowling_team','bowler']][np.logical_and(delivery['over']>=7, delivery['over']<=12)]
temp_bowler_count = temp2.groupby(['bowling_team','bowler'])['bowler'].count()
temp_bowler_count = pd.DataFrame(temp_bowler_count)
temp_bowler_count['temp_1'] = temp_bowler_count.index
temp_bowler_count['team'] = temp_bowler_count['temp_1'].str[0]
temp_bowler_count['player'] = temp_bowler_count['temp_1'].str[1]
temp_bowler_count.drop(['temp_1'], axis=1, inplace=True)
temp_bowler_count.index.name = None
temp_bowler_count = temp_bowler_count.rename(columns={'bowler':'power_play_bowled'})
temp_bowler_count = temp_bowler_count[:-2]


Middle_overs_twelve_eighteen = delivery[np.logical_and(delivery['over']>=7, delivery['over']<=12)]
Middle_overs_twelve_eighteen = Middle_overs_twelve_eighteen.groupby(['bowling_team','bowler'])['total_runs'].sum().reset_index()
Middle_overs_twelve_eighteen_balls = delivery[np.logical_and(delivery['over']>=7, delivery['over']<=12)].groupby(['bowling_team','bowler'])['ball'].count().reset_index()
Middle_overs_twelve_eighteen_wickets = delivery[np.logical_and(delivery['over']>=7, delivery['over']<=12)].groupby(['bowling_team','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()

Middle_overs_twelve_eighteen = Middle_overs_twelve_eighteen.merge(Middle_overs_twelve_eighteen_balls, left_on=['bowling_team','bowler'], right_on=['bowling_team','bowler'], how='outer')
Middle_overs_twelve_eighteen = Middle_overs_twelve_eighteen.merge(Middle_overs_twelve_eighteen_wickets, left_on=['bowling_team','bowler'], right_on=['bowling_team','bowler'], how='outer')

Middle_overs_twelve_eighteen['economy']=Middle_overs_twelve_eighteen['total_runs']*6/Middle_overs_twelve_eighteen['ball']


#### Batting opening ####




#### Death over scatter plots ####

plt.scatter(death_over_stats[death_over_stats['bowling_team']=='CSK']['economy'], death_over_stats[death_over_stats['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='MI']['economy'], death_over_stats[death_over_stats['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='RCB']['economy'], death_over_stats[death_over_stats['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='DD']['economy'], death_over_stats[death_over_stats['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='black')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='GL']['economy'], death_over_stats[death_over_stats['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='orange')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='SRH']['economy'], death_over_stats[death_over_stats['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='green')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='RR']['economy'], death_over_stats[death_over_stats['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='darkblue')


#### Nineteenth over scatter plots ####

plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='CSK']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='MI']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='RCB']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='DD']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='black')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='GL']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='orange')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='SRH']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='green')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='RR']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='darkblue')


#### First over scatter plots ####

plt.scatter(first_over_stats[first_over_stats['bowling_team']=='CSK']['economy'], first_over_stats[first_over_stats['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='MI']['economy'], first_over_stats[first_over_stats['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='RCB']['economy'], first_over_stats[first_over_stats['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='DD']['economy'], first_over_stats[first_over_stats['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='black')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='GL']['economy'], first_over_stats[first_over_stats['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='orange')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='SRH']['economy'], first_over_stats[first_over_stats['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='green')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='RR']['economy'], first_over_stats[first_over_stats['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='darkblue')


#### Power play scatter plots ####
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='CSK']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='MI']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='RCB']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='DD']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='black')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='GL']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='orange')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='SRH']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='green')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='RR']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='darkblue')



#### Middle over 6 - 12 scatter plots ####
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='CSK']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='MI']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='RCB']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='DD']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='GL']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='SRH']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='RR']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='yellow')


#### Middle over 12 - 18 scatter plots ####
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='CSK']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='MI']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='RCB']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='DD']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='GL']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='SRH']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='RR']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='blue')











for i in range(len(temp)):
    temp.iloc[i]=[x for x in temp.iloc[i] if not isinstance(x, int)]

temp = [x for x in temp if not isinstance(x, int)]



#### Analysis of bowlers ####

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
temp_bowler = bowler[['bowler','economy','wickets']].copy()

temp_train = sc_X.fit_transform(temp_bowler_economy[['economy','wickets']])

temp_train = pd.DataFrame(temp_train)

temp_train.columns = temp_train.columns.astype(str)


for col in temp_train.columns:
    print(col)
    
    
temp_bowler_economy = temp_bowler[np.logical_and(np.logical_and(temp_bowler['economy']>6, temp_bowler['economy']<8), temp_bowler['wickets']>50)]
temp_train = temp_train.rename(columns={'0':'economy'})
plt.scatter(temp_train['economy'], temp_train['wickets'])
plt.scatter(temp_bowler_economy['economy'], temp_bowler_economy['wickets'])








##### Perform K-means #####

np.random.seed(200)

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5)
labels=kmeans.fit(temp_train)
temp_train=temp_train.fillna(0)





labels = kmeans.predict(temp_train)
centroid = kmeans.cluster_centers_



centroids = kmeans.cluster_centers_
colmap = {1:'r',2:'g',3:'b', 4:'y', 5:'k'}
fig = plt.figure(figsize=(5,5))
colors=map(lambda x: colmap[x+1], labels)
color1=list(colors)
plt.scatter(temp_train['economy'],temp_train['wickets'],color=color1,alpha=0.5,edgecolor='k')
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
