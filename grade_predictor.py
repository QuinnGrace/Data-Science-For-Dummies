import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder


def main():
	st.title('Mathematics Grade Predictor')

	sex = st.radio(
    "What's the sex of the student?",
    ["Female", "Male"])

	if sex == "Female":
		sex = "F"
	else:
		sex = "M"

	address = st.radio(
    "What area does the student live in?",
    ["Rural", "Urban"])

	if address == "Rural":
		address = "R"
	else:
		address = "U"

	Medu = st.radio(
	"What is the highest level of education the student's mother achieved?",
	["None","Primary School","5th to 9th Grade", "Finished Secondary","Higher Education"])

	if Medu == "None":
		Medu = "0"
	elif Medu == "Primary School":
		Medu = "1"
	elif Medu == "5th to 9th Grade":
		Medu = "2"
	elif Medu == "Finished Secondary":
		Medu = "3"
	else:
		Medu = "4"

	Fedu = st.radio(
	"What is the highest level of education the student's father achieved?",
	["None","Primary School","5th to 9th Grade", "Finished Secondary","Higher Education"])

	if Fedu == "None":
		Fedu = "0"
	elif Fedu == "Primary School":
		Fedu = "1"
	elif Fedu == "5th to 9th Grade":
		Fedu = "2"
	elif Fedu == "Finished Secondary":
		Fedu = "3"
	else:
		Fedu = "4"

	Mjob = st.radio(
	"What does the student's mother do for work?",
	["Teacher","Healthcare-related","Civil Services", "Stay At Home","Other"])

	if Mjob == "Teacher":
		Mjob = "teacher"
	elif Mjob == "Healthcare-related":
		Mjob = "health"
	elif Mjob == "Civil Services":
		Mjob = "services"
	elif Mjob == "Stay At Home":
		Mjob = "at_home"
	else:
		Mjob = "other"

	reason = st.radio(
	"What was the reason for choosing this school?",
	["Close to Home", "School Reputation", "Course Preference", "Other"]
	)

	if reason == "Close to Home":
		reason = "home"
	elif reason == "School Reputation":
		reason = "reputation"
	elif reason == "Course Preference":
		reason = "preference"


	paid = st.radio(
	"Does the student take extra paid classes for Mathematics?",
	["Yes","No"]
	)

	if paid == "Yes":
		paid = "yes"
	else:
		paid = "no"

	higher = st.radio(
	"Does the student want to pursue higher education?",
	["Yes","No"]
	)

	if higher == "Yes":
		higher = "yes"
	else:
		higher = "no"

	result = [sex, address, Medu, Fedu, Mjob, reason, paid, higher]
	result = np.reshape(result, (8, 1)).T
	df = pd.DataFrame(result)

	model_load_path = "encoder.pkl"
	with open(model_load_path,'rb') as file:
		encoder = pickle.load(file)

	result = encoder.transform(df)

	model_load_path = "lr_model.pkl"
	with open(model_load_path,'rb') as file:
		unpickled_model = pickle.load(file)

	prediction = unpickled_model.predict(result)

	st.title("The predicted grade is :")
	st.write(prediction)
	 
	if prediction >= 10:	
		st.image("graduation-hat.png")
		st.write("This student is likely to pass")
	elif prediction >= 8:
		st.image("person.png")
		st.write("This student is at risk of failing")
	else:
		st.image("caution.png")
		st.write("This student is at high risk of failing")

if __name__ == '__main__':
	main()