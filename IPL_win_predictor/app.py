import streamlit as st
import pandas as pd 
import numpy as np 
import pickle as pkl
import time
pipe = pkl.load(open('IPL_model.pkl', 'rb'))

df = pd.read_csv('final_dataset.csv').iloc[:,:-1]
Cities = sorted(df['City'].unique())
Teams = sorted(df['BattingTeam'].unique())

st.title('Welcome to IPL win Predictor')

col1, col2 = st.columns(2)

# Initialize the session state variables
if 'BattingTeam' not in st.session_state:
    st.session_state['BattingTeam'] = Teams[0]  # Initially select the first team in the list

# Create the BattingTeam selectbox
BattingTeam = col1.selectbox('Select Batting Team', Teams, index=Teams.index(st.session_state['BattingTeam']))

# Update the BowlingTeam based on the selected BattingTeam
updated_teams = sorted(set(Teams).difference({BattingTeam}))
BowlingTeam = col2.selectbox('Select Bowling Team', updated_teams)

# Update the session state variable for BattingTeam
st.session_state['BattingTeam'] = BattingTeam

city = st.selectbox('Select City', Cities)

Target = st.number_input('Target',value = 0,  max_value=1000, min_value = 0)
col3, col4, col5 = st.columns(3)

Score = col3.number_input('Score', value = 0,  max_value=Target + 6, min_value = 0)
col4_1, col4_2 = col4.columns(2)

Over = col4_1.number_input('Over Completed',value = 0,  max_value=20, min_value = 0)
Ball = col4_2.number_input('Ball Completed',value = 1,  max_value=5, min_value = 0)
Wickets = col5.number_input('Wickets Gone',value = 0,  max_value=10, min_value = 0)
# st.write(df.columns)
predict = st.button('Predict Probability')

def score(input_df):
    result = pipe.predict_proba(input_df)
    return map(lambda x: round(x*100),result[0])

if predict:
    runs_left = Target - Score
    balls_left = 120 - (Over*6 + Ball)
    wicket_left = 10 - Wickets
    crr = Score / ((Over*6 + Ball)/6)
    rrr = runs_left / (balls_left/6)
    data = [[city, BattingTeam, BowlingTeam, runs_left,balls_left, wicket_left, Target, crr, rrr]]
    input_df = pd.DataFrame(data, columns=list(df.columns))
    col6,col7,col8 = st.columns(3)
    loss, win = score(input_df)
    col6.write(f"{BattingTeam}  \n  {win}%")
    col8.write(f"{BowlingTeam}  \n  {loss}%")
    progress_bar = st.progress(0)
    for i in range(0,win + 1):
        progress_bar.progress(i)
        time.sleep(0.005)

    st.table(input_df[['runs_left','balls_left','crr','rrr']])


    

    






