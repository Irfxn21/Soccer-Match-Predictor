import pandas as pd

matches = pd.read_csv("matches.csv", index_col=0) # Reading match data

# Converting objects into numerical format for machine learning
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes # COnverting venue to home (1)/away (0)
matches["opp_code"] = matches["opponent"].astype("category").cat.codes #Enumerating opponents
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int") #Converting time of match into numerical number
matches["day_code"] = matches["date"].dt.dayofweek #Converting day of week into number (0-6)
matches["target"] = (matches["result"] == "W").astype("int") # Setting win to value of 1, lose and draw to value of 0

#Importing machine learning model for non-linear data
from sklearn.ensemble import RandomForestClassifier 
# Initial model
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01'] # set of matches used to train model
test = matches[matches["date"] > '2022-01-01'] # Set of matches to predict
predictors = ["venue_code", "opp_code", "hour", "day_code"] 
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])

from sklearn.metrics import accuracy_score

acc = accuracy_score(test["target"], preds) ## testing accuracy
#print(acc) ---> uncomment to view initial accuracy
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"]) # Comparing actual and predicted results

from sklearn.metrics import precision_score

# Improving accuracy of model

def rolling_averages(group, cols, new_cols): #Function that takes previous games into consdieration
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) #Dropping missing values and replacing with empty
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols] #Add new columns with rolling averages values

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0]) #Remove index level on team names

def make_predictions(data, predictors): # Function that makes the predictions
    train = data[data["date"] < '2022-01-01']
    test  = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors]) # Make predictions
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision # Returning prediciton values

combined, precision = make_predictions(matches_rolling, predictors + new_cols)

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

class MissingDict(dict): # Class to replace team names with common names
    __missing__ = lambda self, key: key

# Team names and common their respective common names
map_values = {"Brighton and Hove Albion": "Brighton",
              "Manchester United": "Manchester Utd",
              "Newcastle United": "Newcastle Utd",
              "Tottenham Hotspur": "Tottenham",
              "West Ham United": "West Ham",
              "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

combined["new_team"] = combined["team"].map(mapping) # Adding common names

# Finding both home and away results and combining them into one table
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

print(merged) # Print predictions




## project inspired by dataquest tutorial
