# -*- coding: utf-8 -*-
"""
@author: Kshitij Singh
Description: Contains complete analysis of IPL 2008-2017
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
os.chdir(r"<--Directory Path-->") # For macOS
os.chdir(r"<--Directory Path-->") # For windows

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


'''
This section contains basic parameters like:
    names of team
    total matches played
    cities
    venues(stadium)
    umpire names
    man of the matches
'''
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
total_umpires = pd.concat([matches['umpire1'],
                           matches['umpire2']])
total_umpires = total_umpires.value_counts().reset_index()

# Man of the matches
man_of_matches=matches['player_of_match'].value_counts().reset_index()
man_of_matches.columns=['Player Name', 
                        'player_of_match']

# Visualisation of Umpires
total_umpires.columns=['Umpire Name', 
                       'Matches Umpired']
plt.xticks(rotation=90, 
           fontsize=6)
sns.barplot(x='Umpire Name',
            y='Matches Umpired',
            data=total_umpires)

# Visualisation of Man of the match
fig = plt.gcf()
fig.set_size_inches(18.5,
                    10.5)
plt.xticks(rotation=45, 
           fontsize=6)
plt.ylim(0,
         18)
sns.barplot(x='Player Name', 
            y='player_of_match', 
            data=man_of_matches.head(20))


'''
This section is solely for batsman data.
I used the matches dataframe and use feature engineering extract the following parameters for each batsman:
    1. singles
    2. doubles
    3. threes
    4. fours
    5. fives
    6. sixes
    scored.
Additionally, I also calculate their highest scores, man of match won, hundreds, fifties etc.
'''

# Computing and storing batsman data


# Calculating batsman parameters and storing them in batting
batting = delivery.groupby(['batsman'])['ball'].count().reset_index()

# Total runs
runs = delivery.groupby(['batsman'])['batsman_runs'].sum().reset_index() 

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
batting['average'] = batting['runs_scored']/batting['matches']



'''
In this section, I analyse the bowler in depth.
I have segregated the bowlers into the following as:
    1. First over bowlers
    2. Power Play bowlers
    3. Middle Over bowlers
    4. Middle Over bowlers
    5. Nineteenth Over bowlers
    6. Death Over bowlers

I calculate different parameters such as economy, and wickets taken per match f
or each category.
To drill down further, the bowlers are also classified based on their bowling 
experience(number of balls bowled) in each category.
'''


# Calculating bowler parameters including all overs
balls = delivery.groupby(['bowler'])['ball'].count().reset_index()

# Runs conceded
runs_conceded = delivery.groupby(['bowler'])['total_runs'].sum().reset_index()
runs_conceded = runs_conceded.rename(columns={'total_runs':'runs_conceded'})

# Five wicket haul
fifer = delivery.groupby(['match_id','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
fifer = fifer.groupby(['bowler'])['player_dismissed'].agg(lambda x: (x>=5).sum()).reset_index()
fifer = fifer.rename(columns={'player_dismissed':'fifer'})

# Four wicket haul
fours_w = delivery.groupby(['match_id','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
fours_w = fours_w.groupby(['bowler'])['player_dismissed'].agg(lambda x: (x==4).sum()).reset_index()
fours_w = fours_w.rename(columns={'player_dismissed':'fours'})

# Three wicket haul
threes_w = delivery.groupby(['match_id','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
threes_w = threes_w.groupby(['bowler'])['player_dismissed'].agg(lambda x: (x==3).sum()).reset_index()
threes_w = threes_w.rename(columns={'player_dismissed':'threes'})

# Two wicket
twos_w = delivery.groupby(['match_id','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
twos_w = twos_w.groupby(['bowler'])['player_dismissed'].agg(lambda x: (x==2).sum()).reset_index()
twos_w = twos_w.rename(columns={'player_dismissed':'twos'})

# One wicket
ones_w = delivery.groupby(['match_id','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
ones_w = ones_w.groupby(['bowler'])['player_dismissed'].agg(lambda x: (x==1).sum()).reset_index()
ones_w = ones_w.rename(columns={'player_dismissed':'ones'})

# Number of wides
wides = delivery.groupby(['bowler'])['wide_runs'].agg(lambda x: x.sum()).reset_index()

# Number of bye runs
bye_runs = delivery.groupby(['bowler'])['bye_runs'].agg(lambda x: x.sum()).reset_index()

# Number of no balls bowled
no_balls_runs = delivery.groupby(['bowler'])['noball_runs'].agg(lambda x: x.sum()).reset_index()

# Total penalty runs conceded
penalty_runs = delivery.groupby(['bowler'])['penalty_runs'].agg(lambda x: x.sum()).reset_index()

# Total runs conceded as extra runs
extra_runs = delivery.groupby(['bowler'])['extra_runs'].agg(lambda x: x.sum()).reset_index()

# Dismissal by fielder
caught = delivery.groupby(['bowler'])['dismissal_kind'].agg(lambda x: (x=='caught').sum()).reset_index()
caught =  caught.rename(columns={'dismissal_kind':'caught'})

# Dismissal by bowled
bowled = delivery.groupby(['bowler'])['dismissal_kind'].agg(lambda x: (x=='bowled').sum()).reset_index()
bowled = bowled.rename(columns={'dismissal_kind':'bowled'})

# Dismissal by caught and bowled
caught_and_bowled = delivery.groupby(['bowler'])['dismissal_kind'].agg(lambda x: (x=='caught and bowled').sum()).reset_index()
caught_and_bowled = caught_and_bowled.rename(columns={'dismissal_kind':'caught_and_bowled'})

# Dismissal by leg before wicket
lbw =  delivery.groupby(['bowler'])['dismissal_kind'].agg(lambda x: (x=='lbw').sum()).reset_index()
lbw = lbw.rename(columns={'dismissal_kind':'lbw'})

# Number of matches played by each bowler
matches_played_bowler = delivery.groupby(['bowler'])['match_id'].nunique().reset_index()
matches_played_bowler=matches_played_bowler.rename(columns={'match_id':'matches'})


# Merging with bowling
bowler = delivery.groupby(['bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()
bowler = bowler.rename(columns={'player_dismissed':'wickets'})
bowler = bowler.merge(balls, left_on='bowler', right_on='bowler', how='outer')
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
bowler = bowler.merge(matches_played_bowler, left_on='bowler', right_on='bowler', how='outer')


# Calculating economy 
# Economy is runs conceded by the bowler per six balls bowled
# Wicket economy is the average number of wickets taken by bowler in each match

bowler['economy'] = bowler['runs_conceded']*6/(bowler['ball'])
bowler['wicket_economy'] = bowler['wickets']/bowler['matches']


'''
Each section begins with the name of the classification.

In each section, we have some temporary variables identified by temp 
followed by some self explanatory suffix.

Finally, all the temporary variables are dropped.

'''

############################## Death Over Bowlers ##############################

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
death_over_wickets = delivery[delivery['over']==20].groupby(['bowling_team','bowler'])['player_dismissed'].agg(lambda x:(x!=0).sum()).reset_index()



death_over_stats = death_over_stats.merge(death_over_balls, left_on=['bowler','bowling_team'], right_on=['bowler', 'bowling_team'], how='outer')
death_over_stats = death_over_stats.merge(death_over_wickets, left_on=['bowler','bowling_team'], right_on=['bowler','bowling_team'], how='outer')
temp_bowler_count = temp_bowler_count.merge(death_over_stats['total_runs'], left_on='player', right_on='bowler', how='outer')
death_over_stats['economy'] = death_over_stats['total_runs']*6/death_over_stats['ball']

temp_bowler_count = temp_bowler_count.rename(columns={'team':'bowling_team','player':'bowler'}).reset_index()
temp_bowler_count.index.name = None
death_over_stats.index.name = None
death_over_stats = death_over_stats.merge(temp_bowler_count, left_on='bowler', right_on='bowler', how='outer')
death_over_stats.drop('ball_x', axis=1, inplace=True)
death_over_stats.drop('player_dismissed_x', axis=1, inplace=True)
death_over_stats = death_over_stats.rename(columns={'player_dismissed_y':'player_dismissed'})

death_over_played_bowler = delivery[delivery['over']==20].groupby(['bowler'])['match_id'].nunique().reset_index()
death_over_played_bowler=death_over_played_bowler.rename(columns={'match_id':'matches'})
death_over_stats = death_over_stats.merge(death_over_played_bowler, left_on='bowler', right_on='bowler', how='outer')


death_over_stats['wicket_economy'] = death_over_stats['player_dismissed']/death_over_stats['matches']

# Dropping unnecessary parameters
del temp_bowler_count
del temp1
del temp2
del death_over_balls
del death_over_bowlers
del death_over_wickets
del death_over_played_bowler

############################## 19th Over Bowlers ##############################
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
nineteenth_over_played_bowler = delivery[delivery['over']==19].groupby(['bowler'])['match_id'].nunique().reset_index()
nineteenth_over_played_bowler=nineteenth_over_played_bowler.rename(columns={'match_id':'matches'})
nineteenth_over_stats = nineteenth_over_stats.merge(nineteenth_over_played_bowler, left_on='bowler', right_on='bowler', how='outer')

nineteenth_over_stats['wicket_economy'] = nineteenth_over_stats['player_dismissed']/nineteenth_over_stats['matches']

# Dropping unnecessary parameters
del temp_bowler_count
del temp2
del nineteenth_over_balls
del nineteenth_over_wickets
del nineteenth_over_played_bowler

############################## 1st Over Bowlers ##############################
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

first_over_played_bowler = delivery[delivery['over']==1].groupby(['bowler'])['match_id'].nunique().reset_index()
first_over_played_bowler=first_over_played_bowler.rename(columns={'match_id':'matches'})
first_over_stats = first_over_stats.merge(first_over_played_bowler, left_on='bowler', right_on='bowler', how='outer')


first_over_stats['wicket_economy'] = first_over_stats['player_dismissed']/first_over_stats['matches']

# Dropping unnecessary parameters
del temp_bowler_count
del temp2
del temp
del first_over_balls
del first_over_wickets
del first_over_played_bowler

############################## Power Play Over Bowlers ##############################s
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

power_play_over_played_bowler = delivery[delivery['over']<=6].groupby(['bowler'])['match_id'].nunique().reset_index()
power_play_over_played_bowler=power_play_over_played_bowler.rename(columns={'match_id':'matches'})
power_play_bowling = power_play_bowling.merge(power_play_over_played_bowler, left_on='bowler', right_on='bowler', how='outer')


power_play_bowling['wicket_economy'] = power_play_bowling['player_dismissed']/power_play_bowling['matches']


# Dropping unnecessary parameters
del temp_bowler_count
del temp2
del power_play_bowling_balls
del power_play_bowling_wickets
del power_play_over_played_bowler

############################## Middle Over (6-12) Bowlers ##############################
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

MOST_over_played_bowler = delivery[np.logical_and(delivery['over']>=7, delivery['over']<=12)].groupby(['bowler'])['match_id'].nunique().reset_index()
MOST_over_played_bowler=MOST_over_played_bowler.rename(columns={'match_id':'matches'})
Middle_overs_six_twelve = Middle_overs_six_twelve.merge(MOST_over_played_bowler, left_on='bowler', right_on='bowler', how='outer')


Middle_overs_six_twelve['wicket_economy'] = Middle_overs_six_twelve['player_dismissed']/Middle_overs_six_twelve['matches']

# Dropping unnecessary parameters
del temp_bowler_count
del temp2
del Middle_overs_six_twelve_balls
del Middle_overs_six_twelve_wickets
del MOST_over_played_bowler

############################## Middle Over (12-18) Bowlers ##############################
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

MOTE_over_played_bowler = delivery[np.logical_and(delivery['over']>=7, delivery['over']<=12)].groupby(['bowler'])['match_id'].nunique().reset_index()
MOTE_over_played_bowler=MOTE_over_played_bowler.rename(columns={'match_id':'matches'})
Middle_overs_twelve_eighteen = Middle_overs_twelve_eighteen.merge(MOTE_over_played_bowler, left_on='bowler', right_on='bowler', how='outer')


Middle_overs_twelve_eighteen['wicket_economy'] = Middle_overs_twelve_eighteen['player_dismissed']/Middle_overs_twelve_eighteen['matches']


# Dropping unnecessary parameters
del temp_bowler_count
del temp2
del Middle_overs_twelve_eighteen_balls
del Middle_overs_twelve_eighteen_wickets
del MOTE_over_played_bowler



'''
In this section, we use K-Means algorithm to cluster similar bowlers 
within each classification and sub classification(bowler experience)

Copy the variables and then perform normalization on them.
Minimax normalization has been used.

'''

############################## Normalizing Bowlers ##############################

first_over= pd.DataFrame()
power_player = pd.DataFrame()
middle_over_s_t = pd.DataFrame()
middle_over_t_e = pd.DataFrame()
nineteenth = pd.DataFrame()
death_over = pd.DataFrame()

first_over = first_over_stats.groupby('bowler')['ball'].agg(lambda x:x.sum()).reset_index()
temp = first_over_stats.groupby('bowler')['total_runs'].agg(lambda x:x.sum()).reset_index()
temp1 = first_over_stats.groupby('bowler')['matches'].agg(lambda x:x.sum()).reset_index()
temp2 = first_over_stats.groupby('bowler')['player_dismissed'].agg(lambda x:x.sum()).reset_index()

first_over = first_over.merge(temp, left_on='bowler', right_on='bowler', how='outer')
first_over = first_over.merge(temp1, left_on='bowler', right_on='bowler', how='outer')
first_over = first_over.merge(temp2, left_on='bowler', right_on='bowler', how='outer')


first_over['economy']=first_over['total_runs']*6/first_over['ball']
first_over['wicket_economy']=first_over['player_dismissed']/first_over['matches']



power_player = power_play_bowling.groupby('bowler')['ball'].agg(lambda x:x.sum()).reset_index()
temp = power_play_bowling.groupby('bowler')['total_runs'].agg(lambda x:x.sum()).reset_index()
temp1 = power_play_bowling.groupby('bowler')['matches'].agg(lambda x:x.sum()).reset_index()
temp2 = power_play_bowling.groupby('bowler')['player_dismissed'].agg(lambda x:x.sum()).reset_index()

power_player = power_player.merge(temp, left_on='bowler', right_on='bowler', how='outer')
power_player = power_player.merge(temp1, left_on='bowler', right_on='bowler', how='outer')
power_player = power_player.merge(temp2, left_on='bowler', right_on='bowler', how='outer')


power_player['economy']=power_player['total_runs']*6/power_player['ball']
power_player['wicket_economy']=power_player['player_dismissed']/power_player['matches']


middle_over_s_t = Middle_overs_six_twelve.groupby('bowler')['ball'].agg(lambda x:x.sum()).reset_index()
temp = Middle_overs_six_twelve.groupby('bowler')['total_runs'].agg(lambda x:x.sum()).reset_index()
temp1 = Middle_overs_six_twelve.groupby('bowler')['matches'].agg(lambda x:x.sum()).reset_index()
temp2 = Middle_overs_six_twelve.groupby('bowler')['player_dismissed'].agg(lambda x:x.sum()).reset_index()

middle_over_s_t = middle_over_s_t.merge(temp, left_on='bowler', right_on='bowler', how='outer')
middle_over_s_t = middle_over_s_t.merge(temp1, left_on='bowler', right_on='bowler', how='outer')
middle_over_s_t = middle_over_s_t.merge(temp2, left_on='bowler', right_on='bowler', how='outer')


middle_over_s_t['economy']=middle_over_s_t['total_runs']*6/middle_over_s_t['ball']
middle_over_s_t['wicket_economy']=middle_over_s_t['player_dismissed']/middle_over_s_t['matches']


middle_over_t_e = Middle_overs_twelve_eighteen.groupby('bowler')['ball'].agg(lambda x:x.sum()).reset_index()
temp = Middle_overs_twelve_eighteen.groupby('bowler')['total_runs'].agg(lambda x:x.sum()).reset_index()
temp1 = Middle_overs_twelve_eighteen.groupby('bowler')['matches'].agg(lambda x:x.sum()).reset_index()
temp2 = Middle_overs_twelve_eighteen.groupby('bowler')['player_dismissed'].agg(lambda x:x.sum()).reset_index()

middle_over_t_e = middle_over_t_e.merge(temp, left_on='bowler', right_on='bowler', how='outer')
middle_over_t_e = middle_over_t_e.merge(temp1, left_on='bowler', right_on='bowler', how='outer')
middle_over_t_e = middle_over_t_e.merge(temp2, left_on='bowler', right_on='bowler', how='outer')


middle_over_t_e['economy']=middle_over_t_e['total_runs']*6/middle_over_t_e['ball']
middle_over_t_e['wicket_economy']=middle_over_t_e['player_dismissed']/middle_over_t_e['matches']


nineteenth = nineteenth_over_stats.groupby('bowler')['ball'].agg(lambda x:x.sum()).reset_index()
temp = nineteenth_over_stats.groupby('bowler')['total_runs'].agg(lambda x:x.sum()).reset_index()
temp1 = nineteenth_over_stats.groupby('bowler')['matches'].agg(lambda x:x.sum()).reset_index()
temp2 = nineteenth_over_stats.groupby('bowler')['player_dismissed'].agg(lambda x:x.sum()).reset_index()

nineteenth = nineteenth.merge(temp, left_on='bowler', right_on='bowler', how='outer')
nineteenth = nineteenth.merge(temp1, left_on='bowler', right_on='bowler', how='outer')
nineteenth = nineteenth.merge(temp2, left_on='bowler', right_on='bowler', how='outer')


nineteenth['economy']=nineteenth['total_runs']*6/nineteenth['ball']
nineteenth['wicket_economy']=nineteenth['player_dismissed']/nineteenth['matches']

death_over = death_over_stats.groupby('bowler')['ball'].agg(lambda x:x.sum()).reset_index()
temp = death_over_stats.groupby('bowler')['total_runs'].agg(lambda x:x.sum()).reset_index()
temp1 = death_over_stats.groupby('bowler')['matches'].agg(lambda x:x.sum()).reset_index()
temp2 = death_over_stats.groupby('bowler')['player_dismissed'].agg(lambda x:x.sum()).reset_index()

death_over = death_over.merge(temp, left_on='bowler', right_on='bowler', how='outer')
death_over = death_over.merge(temp1, left_on='bowler', right_on='bowler', how='outer')
death_over = death_over.merge(temp2, left_on='bowler', right_on='bowler', how='outer')


death_over['economy']=death_over['total_runs']*6/death_over['ball']
death_over['wicket_economy']=death_over['player_dismissed']/death_over['matches']

# Normalizing
# w_economy - stores the wicket economy(Wickets taken per match)
# economy - - stores the runs economy(Runs conceded per 6 balls)
from sklearn.preprocessing import  MinMaxScaler
sc= MinMaxScaler()

# First Over
economy = first_over['economy'].values.astype(float)
economy=economy.reshape(-1,1)
w_economy = first_over['wicket_economy'].values.astype(float)
w_economy=w_economy.reshape(-1,1)
economy= sc.fit_transform(economy)
w_economy = sc.fit_transform(w_economy)

economy = pd.DataFrame(economy)
w_economy = pd.DataFrame(w_economy)

first_over['economy_norm'] = economy
first_over['w_economy_norm'] = w_economy



# Power Play
economy = power_player['economy'].values.astype(float)
economy=economy.reshape(-1,1)
w_economy = power_player['wicket_economy'].values.astype(float)
w_economy=w_economy.reshape(-1,1)
economy= sc.fit_transform(economy)
w_economy = sc.fit_transform(w_economy)

economy = pd.DataFrame(economy)
w_economy = pd.DataFrame(w_economy)

power_player['economy_norm'] = economy
power_player['w_economy_norm'] = w_economy


# Middle over (6-12)
economy = middle_over_s_t['economy'].values.astype(float)
economy=economy.reshape(-1,1)
w_economy = middle_over_s_t['wicket_economy'].values.astype(float)
w_economy=w_economy.reshape(-1,1)
economy= sc.fit_transform(economy)
w_economy = sc.fit_transform(w_economy)

economy = pd.DataFrame(economy)
w_economy = pd.DataFrame(w_economy)

middle_over_s_t['economy_norm'] = economy
middle_over_s_t['w_economy_norm'] = w_economy


# Middle over(12-18)
economy = middle_over_t_e['economy'].values.astype(float)
economy=economy.reshape(-1,1)
w_economy = middle_over_t_e['wicket_economy'].values.astype(float)
w_economy=w_economy.reshape(-1,1)
economy= sc.fit_transform(economy)
w_economy = sc.fit_transform(w_economy)

economy = pd.DataFrame(economy)
w_economy = pd.DataFrame(w_economy)

middle_over_t_e['economy_norm'] = economy
middle_over_t_e['w_economy_norm'] = w_economy


# Nineteenth over
economy = nineteenth['economy'].values.astype(float)
economy=economy.reshape(-1,1)
w_economy = nineteenth['wicket_economy'].values.astype(float)
w_economy=w_economy.reshape(-1,1)
economy= sc.fit_transform(economy)
w_economy = sc.fit_transform(w_economy)

economy = pd.DataFrame(economy)
w_economy = pd.DataFrame(w_economy)

nineteenth['economy_norm'] = economy
nineteenth['w_economy_norm'] = w_economy


# Death over
economy = death_over['economy'].values.astype(float)
economy=economy.reshape(-1,1)
w_economy = death_over['wicket_economy'].values.astype(float)
w_economy=w_economy.reshape(-1,1)
economy= sc.fit_transform(economy)
w_economy = sc.fit_transform(w_economy)

economy = pd.DataFrame(economy)
w_economy = pd.DataFrame(w_economy)

death_over['economy_norm'] = economy
death_over['w_economy_norm'] = w_economy


######################### Finding Optimium Clustering #########################
'''
In this section, k-means is applied for each classification and 
sub-classification.
'''
#### First over ####
####### Balls >= 100 #######
# Optimum K for balls >= 100
from sklearn.cluster import KMeans
cost =[] 

for i in range(1, 20): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(first_over[first_over['ball']>=100][['economy_norm','w_economy_norm']])
    cost.append(KM.inertia_)     

plt.plot(cost)

# Clustering for balls >= 100
KM = KMeans(n_clusters = 4, max_iter = 500) 
KM.fit(first_over[first_over['ball']>=100][['economy_norm','w_economy_norm']]) 
KM.labels_
plt.scatter(first_over[first_over['ball']>=100]['economy_norm'], first_over[first_over['ball']>=100]['w_economy_norm'], c=KM.labels_, cmap='rainbow')


####### Balls >= 70 and <= 100 #######
# Optimum K for balls >= 70 and <=100
from sklearn.cluster import KMeans
cost =[] 

for i in range(1, 20): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(first_over[np.logical_and(first_over['ball']>=50,first_over['ball']<100)][['economy_norm','w_economy_norm']])
    cost.append(KM.inertia_)     

plt.plot(cost)

# Clustering for balls >= 100
KM = KMeans(n_clusters = 3, max_iter = 500) 
KM.fit(first_over[np.logical_and(first_over['ball']>=50,first_over['ball']<100)][['economy_norm','w_economy_norm']]) 
KM.labels_
plt.scatter(first_over[np.logical_and(first_over['ball']>=50,first_over['ball']<100)]['economy_norm'], first_over[np.logical_and(first_over['ball']>=50,first_over['ball']<100)]['w_economy_norm'], c=KM.labels_, cmap='rainbow')


####### Balls >= 6 and <= 50 #######
# Optimum K for balls >= 6 and <= 50
from sklearn.cluster import KMeans
cost =[] 

for i in range(1, 20): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(first_over[np.logical_and(first_over['ball']>=20,first_over['ball']<50)][['economy_norm','w_economy_norm']])
    cost.append(KM.inertia_)     

plt.plot(cost)

# Clustering for balls >= 100
KM = KMeans(n_clusters = 3, max_iter = 500) 
KM.fit(first_over[np.logical_and(first_over['ball']>=20,first_over['ball']<50)][['economy_norm','w_economy_norm']]) 
KM.labels_
plt.scatter(first_over[np.logical_and(first_over['ball']>=20,first_over['ball']<50)]['economy_norm'], first_over[np.logical_and(first_over['ball']>=20,first_over['ball']<50)]['w_economy_norm'], c=KM.labels_, cmap='rainbow')


#### power play overs ####
#### Balls <= 650 and >=300
from sklearn.cluster import KMeans
cost =[] 

for i in range(1, 20): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(power_player[np.logical_and(power_player['ball']>=300, power_player['ball']<=650)][['economy_norm','w_economy_norm']]) 

    cost.append(KM.inertia_)     

KM = KMeans(n_clusters = 5, max_iter = 500) 
KM.fit(power_player[np.logical_and(power_player['ball']>=300, power_player['ball']<=650)][['economy_norm','w_economy_norm']]) 
KM.labels_
plt.scatter(power_player[np.logical_and(power_player['ball']>=300, power_player['ball']<=650)]['economy_norm'], power_player[np.logical_and(power_player['ball']>=200, power_player['ball']<=650)]['w_economy_norm'], c=KM.labels_, cmap='rainbow')

plt.scatter(power_player['economy_norm'], power_player['w_economy_norm'])

#### Balls >= 650
from sklearn.cluster import KMeans
cost =[] 

for i in range(1, 20): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(power_player[power_player['ball']>=650][['economy_norm','w_economy_norm']]) 

    cost.append(KM.inertia_)

KM = KMeans(n_clusters = 2, max_iter = 500) 
KM.fit(power_player[power_player['ball']>=650][['economy_norm','w_economy_norm']]) 
KM.labels_
plt.scatter(power_player[power_player['ball']>=650]['economy_norm'], power_player[power_player['ball']>=650]['w_economy_norm'], c=KM.labels_, cmap='rainbow')

#### Middle Over 6-12
#### Balls <= 650 and >=300
from sklearn.cluster import KMeans
cost =[] 

for i in range(1, 20): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(power_player[np.logical_and(power_player['ball']>=300, power_player['ball']<=650)][['economy_norm','w_economy_norm']]) 

    cost.append(KM.inertia_)     

KM = KMeans(n_clusters = 5, max_iter = 500) 
KM.fit(power_player[np.logical_and(power_player['ball']>=300, power_player['ball']<=650)][['economy_norm','w_economy_norm']]) 
KM.labels_
plt.scatter(power_player[np.logical_and(power_player['ball']>=300, power_player['ball']<=650)]['economy_norm'], power_player[np.logical_and(power_player['ball']>=200, power_player['ball']<=650)]['w_economy_norm'], c=KM.labels_, cmap='rainbow')

plt.scatter(power_player['economy_norm'], power_player['w_economy_norm'])


'''
Clustering may not be suitable for every cluster and hence it is applied only 
to selected classfications.
'''

    
'''
This section contains visualizations of economy of bowlers vs players dismissal count
'''

#### Death over scatter plots ####

plt.scatter(death_over_stats[death_over_stats['bowling_team']=='CSK']['economy'], death_over_stats[death_over_stats['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='MI']['economy'], death_over_stats[death_over_stats['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='RCB']['economy'], death_over_stats[death_over_stats['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='DD']['economy'], death_over_stats[death_over_stats['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='m')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='GL']['economy'], death_over_stats[death_over_stats['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='orange')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='SRH']['economy'], death_over_stats[death_over_stats['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='k')
plt.scatter(death_over_stats[death_over_stats['bowling_team']=='RR']['economy'], death_over_stats[death_over_stats['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='g')


#### Nineteenth over scatter plots ####

plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='CSK']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='MI']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='RCB']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='DD']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='m')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='GL']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='orange')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='SRH']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='k')
plt.scatter(nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='RR']['economy'], nineteenth_over_stats[nineteenth_over_stats['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='g')


#### First over scatter plots ####

plt.scatter(first_over_stats[first_over_stats['bowling_team']=='CSK']['economy'], first_over_stats[first_over_stats['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='MI']['economy'], first_over_stats[first_over_stats['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='RCB']['economy'], first_over_stats[first_over_stats['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='DD']['economy'], first_over_stats[first_over_stats['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='m')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='GL']['economy'], first_over_stats[first_over_stats['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='orange')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='SRH']['economy'], first_over_stats[first_over_stats['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='k')
plt.scatter(first_over_stats[first_over_stats['bowling_team']=='RR']['economy'], first_over_stats[first_over_stats['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='g')


#### Power play scatter plots ####
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='CSK']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='MI']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='RCB']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='DD']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='m')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='GL']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='orange')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='SRH']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='k')
plt.scatter(power_play_bowling[power_play_bowling['bowling_team']=='RR']['economy'], power_play_bowling[power_play_bowling['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='g')



#### Middle over 6 - 12 scatter plots ####
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='CSK']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='MI']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='RCB']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='DD']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='m')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='GL']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='orange')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='SRH']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='k')
plt.scatter(Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='RR']['economy'], Middle_overs_six_twelve[Middle_overs_six_twelve['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='g')


#### Middle over 12 - 18 scatter plots ####
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='CSK']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='CSK']['player_dismissed'], alpha=0.5, color='yellow')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='MI']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='MI']['player_dismissed'], alpha=0.5, color='blue')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='RCB']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='RCB']['player_dismissed'], alpha=0.5, color='red')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='DD']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='DD']['player_dismissed'], alpha=0.5, color='m')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='GL']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='GL']['player_dismissed'], alpha=0.5, color='orange')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='SRH']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='SRH']['player_dismissed'], alpha=0.5, color='k')
plt.scatter(Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='RR']['economy'], Middle_overs_twelve_eighteen[Middle_overs_twelve_eighteen['bowling_team']=='RR']['player_dismissed'], alpha=0.5, color='g')

#### Batting plots
plt.bar(batting[batting['balls_played']>2000]['batsman'], batting[batting['balls_played']>2000]['ones'])
plt.xticks(rotation=45, fontsize=14)
plt.ylable('runs scored')
plt.xlabel('batsman')
plt.title('Singles scored by batsmans')

plt.bar(batting[batting['balls_played']>2000]['batsman'], batting[batting['balls_played']>2000]['twos'])
plt.xticks(rotation=45, fontsize=14)
plt.ylable('runs scored')
plt.xlabel('batsman')
plt.title('Twos scored by batsmans')


plt.bar(batting[batting['balls_played']>2000]['batsman'], batting[batting['balls_played']>2000]['threes'])
plt.xticks(rotation=45, fontsize=14)
plt.ylable('runs scored')
plt.xlabel('batsman')
plt.title('Threes scored by batsmans')


plt.bar(batting[batting['balls_played']>2000]['batsman'], batting[batting['balls_played']>2000]['fours'])
plt.xticks(rotation=45, fontsize=14)
plt.ylable('runs scored')
plt.xlabel('batsman')
plt.title('Fours scored by batsmans')


plt.bar(batting[batting['balls_played']>2000]['batsman'], batting[batting['balls_played']>2000]['fives'])
plt.xticks(rotation=45, fontsize=14)
plt.ylable('runs scored')
plt.xlabel('batsman')
plt.title('Fives scored by batsmans')


plt.bar(batting[batting['balls_played']>2000]['batsman'], batting[batting['balls_played']>2000]['sixes'])
plt.xticks(rotation=45, fontsize=14)
plt.ylable('runs scored')
plt.xlabel('batsman')
plt.title('Sixes scored by batsmans')


plt.bar(batting[batting['balls_played']>2000]['batsman'], batting[batting['balls_played']>2000]['highest'])
plt.xticks(rotation=45, fontsize=14)
plt.ylable('runs scored')
plt.xlabel('batsman')
plt.title('Highest runs scored by batsmans')


plt.bar(batting[batting['balls_played']>2000]['batsman'], batting[batting['balls_played']>2000]['average'])
plt.xticks(rotation=45, fontsize=14)
plt.ylable('runs scored')
plt.xlabel('batsman')
plt.title('Average runs scored by batsmans')


plt.bar(batting[batting['balls_played']>2000]['batsman'], batting[batting['balls_played']>2000]['strike_rate'])
plt.xticks(rotation=45, fontsize=14)
plt.ylable('runs scored')
plt.xlabel('batsman')
plt.title('Strike rate of each batsmans')

# Likewise other plots can also be used for comparision of batsmen.

'''
A bit of teams wise stats for fun.
'''
# Finding out number of toss wins, match wins and matches played overall
#### Team wise victory ####
wins = matches['winner']
toss_wins = matches['toss_winner']

wins = pd.DataFrame(wins)
wins = matches['winner'].value_counts().reset_index()
matches_played = pd.concat([matches['team1'], matches['team2']])
matches_played = pd.DataFrame(matches_played)
matches_played = matches_played.value_counts().reset_index()

matches_played= matches_played.astype(str)
wins= wins.astype(str)

matches_played = matches_played.rename(columns={0:'matches_played'})

wins = wins.merge(matches_played, left_on='teams', right_on='teams', how = 'outer').reset_index()
wins = wins.reset_index()
wins = wins.set_index('teams')

toss_won = matches['toss_winner'].value_counts().reset_index()
toss_won = toss_won.rename(columns={'index':'teams'})
wins = wins.merge(toss_won, left_on='teams', right_on='teams', how = 'outer').reset_index()
