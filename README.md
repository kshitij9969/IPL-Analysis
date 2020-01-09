# IPL Player Analysis For Auctions

Problem Statement - Indian Premier League (IPL) is the largest domestic cricket tournament in the world. A better pool of team would boost the revenue of any tournament. Analyse the players for making better decisions at the auctions.

A good pool of balanced team will directly impact the revenue and retain the trust of stakeholders.
Major sources of revenue are, but not limited to,
1. Sponsorship - ~60% with official partner costing 300$ million
2. Broadcasting Rights - ~ 2.5$ billions
3. Team Sponsors - ~ 400 crore INR
4. Ticket Sales - ~ 200 crore INR
5. Merchandise - ~2 billion

There are two data sets containing ball-by-ball details and match wise details from IPL 2008 to IPL 2017

Major Highlights:
1. Cleansed and preprocessed the dataset using pandas library.
2. Using feature engineering created different parameters like strike rate, economy, wickets taken, matches played, team-wise analysis, win and loss trends and their relation with the decision made after winning the toss.
3. Using K-means clusters similar players into different classes. This would give us more economic options at the time of auctions.
4. For the most effective communication, a Tableau dashboard is created. In this dashboard, players are further sieved through different filters for better player selection.

Conclusion and Future Work - The dashboard and classification helps us understand each player in detail. As future work, I would like to incorporate recent(past 6 months performance in other tournaments) performance of the player which will help us understand the form of the player. A recommendation system which would give suggestion based on the type of player needed in the team.

For most effective communication, a tableau dashboard is created and you can find it here:
https://public.tableau.com/profile/kshitij6814#!/vizhome/IPL_Analysis_15776535832250/Batting
