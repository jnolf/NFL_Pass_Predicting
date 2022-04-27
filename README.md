---- 
## <u>Predicting Pass or Run for NFL Offenses<u>
#### Reported by Jerry Nolf  -  April 27, 2022
---- 
### 1. Overview Project Goals
- Create a model that predicts a pass better than the baseline
- Identify key drivers for an offensive play resulting in a pass
- Incorporate clustering to refine model
- Construct a machine learning regression classification model
- Deliver a report for the Defensive Coordinator with findings


---- 
### 2. Project Description
The NFL has been one of the largest organizations to introduce analytics into its multi-faceted operation and has made available very useful information that can help in better understanding the ever-changing game of American football.

This project is a scenario that a Defensive Coordinator has asked for a report on how offensive mindsets have changed over the last 5 seasons and whether or not available data can be used in order to better predict offensive play outcomes.

In order to provide an idea of the impact NFL data has, I will be taking play-by-play data from the past 5 seasons (2017-2021) and using it in order to predict whether an offensive play will be will result in a pass or a run.


---- 
### 3. Initial Questions/Hypothesis
- Has passing increased over the years?
- Is there a relationship between what down it is and passing the ball?
- Is there a linear relationship between current quarter and passing the ball?
- Has offensive produced yards increased over time?


---- 
### 4. Data Dictionary 
| Column            | Description                    | Data Type  |               
|-------------------|--------------------------------|------------|           
|GameDate           | Date game was played           |object      |                
|Quarter            | Quarter the play happened      |int64       |                
|Minute             | Minute the play happened       |int64       |                
|Second             | Second the play happened       |int64       |              
|OffenseTeam        | Team playing offense           |object      |           
|DefenseTeam        | Team playing defense           |object      |           
|Down               | What down the play occured on  |int64       |           
|ToGo               | Yards until a 1st down         |int64       |           
|YardLine           | Yards needed to score a touchdown  |int64   |           
|SeriesFirstDown    | Whether play is first of drive |int64       |           
|SeasonYear         | Season that play occured       |int64       |           
|Yards              | How many yards gain on play    |int64       |           
|Formation          | Formation of offense           |object      |           
|PlayType           | Type of play performed         |object      |           
|IsPass             | Was the play a pass            |int64       |           
|IsInterception     | Was the play an interception   |int64       |           
|IsFumble           | Was the play a fumble          |int64       |           
|YardLineFixed      | Yardline the play started on   |int64       |           
|YardLineDirection  | Side of field play started     |object      |           
|YTG_bins           | Yards to go bins used          |category    |           
|QuarterSeconds     | Maximum seconds up to current quarter  |int64  |           
|ClockSeconds       | Total seconds on the game clock |int64       |           
|SecondsLeft        | Total seconds left in the game |int64        | 


---- 
## PROCESS:
The following outlines the process taken through the Data Science Pipeline to complete this project.  

Plan ➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver

### 1. PLAN
- Define the project goal
- Determine proper format for the audience (Defensive Coordinator)
- Asked questions that would lead to final goal
- Define an MVP


### 2. ACQUIRE
- Create a function to pull appropriate information from the play-by-play csv files for 2017-2021 seasons
- Create and save a wrangle.py file in order to use the function to acquire


### 3. PREPARE
- Ensure all data types are usable
- Create a function that  will:
        - drop unneeded columns
        - get rid of kick formation rows 
        - get rid of other plays other than pass and run
        - create a 'SecondsLeft' column
- Add a function that splits the acquired data into Train, Validate, and Test sets
- 20% is originally pulled out in order to test in the end
- From the remaining 80%, 30% is pullout out to validate training
- The remaining data is used as testing data
- In the end, there should be a 56% Train, 24% Validate, and 20% Test split 


### 4. EXPLORE
- Create an exploratory workbook
- Create initial questions to help explore the data further
- Make visualizations to help identify and understand key driver
- Create clusters in order to dive deeper and refine features
- Use stats testing on established hypotheses


### 5. MODEL & EVALUATE
- Use clusters to evaluate drivers of assessed tax value
- Create a baseline
- Make predictions of models and what they say about the data
- Compare all models to evaluate the best for use
- Use the best performing model on the test (unseen data) sample
- Compare the modeled test versus the baseline


### 6. DELIVERY
- Present a final Jupyter Notebook
- Make modules used and project files available on Github

---- 
## REPRODUCIBILITY: 
	
### Steps to Reproduce
1. Have your env file with proper credentials saved to the working directory

2. Ensure that a .gitignore is properly made in order to keep privileged information private

3. Clone repo from github to ensure availability of the acquire and prepare imports

4. Ensure pandas, numpy, matplotlib, scipy, sklearn, and seaborn are available

5. Follow steps outline in this README.md to run Final_Zillow_Report.ipynb


---- 
## KEY TAKEAWAYS:

### Conclusion:
#### The goals of this project were to identify drivers of pass plays in order to predict whether or not a play would result in a pass or run for NFL offenses. Key drivers found were the following:

- Down
- Yardline
- togo_cluster (Yards to go along with seconds left in the game)
Using these drivers to help our model resulted in an increase of 3% over the baseline.

#### Recommendation(s):
An increase of 2.63% isn't impactful enough to push the model forward to help with predictions in future or real time scenarios. The model must be further refined.

#### Next Steps:
With more time, I would like to:

- Work on more feature engineering and explore more relationships of categories to passing and/or running the ball.

- Explore other datasets to find and create features that will help refine our current model or result in the creation of a new model.

- Look into focusing on plays that happened on specific down and by specific teams. 

---- 
