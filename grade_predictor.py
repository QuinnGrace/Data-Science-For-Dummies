import streamlit as st # Website building
import pandas as pd # DataFrame handling
import numpy as np # Data manipulation
import pickle # Importing ready models
import sklearn # Breaks without this for some reason
from sklearn.preprocessing import OneHotEncoder # Creates dummies

# Create the app
def main():
	# Heading
	st.title('Mathematics Grade Predictor')

	# Buttons to select options
	age = st.radio(
    "What's the age of the student?",
    [15, 16, 17, 18, 19, 20, 21, 22])
	
	Medu = st.radio(
	"What is the highest level of education the student's mother achieved?",
	["None","Primary School","5th to 9th Grade", "Finished Secondary","Higher Education"])
	# Changing the data to a format the model will recognize
	if Medu == "None":
		Medu = 0
	elif Medu == "Primary School":
		Medu = 1
	elif Medu == "5th to 9th Grade":
		Medu = 2
	elif Medu == "Finished Secondary":
		Medu = 3
	else:
		Medu = 4

	Fedu = st.radio(
	"What is the highest level of education the student's father achieved?",
	["None","Primary School","5th to 9th Grade", "Finished Secondary","Higher Education"])

	if Fedu == "None":
		Fedu = 0
	elif Fedu == "Primary School":
		Fedu = 1
	elif Fedu == "5th to 9th Grade":
		Fedu = 2
	elif Fedu == "Finished Secondary":
		Fedu = 3
	else:
		Fedu = 4

	failures = st.radio(
	"How many times has the student failed the class?",
	[0,1,2,3,4]
	)

	higher = st.radio(
	"Does the student want to pursue higher education?",
	["Yes","No"]
	)

	if higher == "Yes":
		higher = "yes"
	else:
		higher = "no"
	
	# Collecting all the chosen options into a list
	result = [age, Medu, Fedu, failures, higher]
	# Preparing the list to be turned into a dataframe
	result = np.reshape(result, (5, 1)).T
	# Turning the list into a dataframe
	df = pd.DataFrame(result, columns=['age', 'Medu', 'Fedu', 'failures','higher'])
	# "unpickling" (opening) the preprocessor
	model_load_path = "encoder.pkl"
	with open(model_load_path,'rb') as file:
		encoder = pickle.load(file)

	# Using the preprocessor on our list
	preproc = encoder.transform(df)
	# "unpickling" (opening) the trained linear model
	model_load_path = "lr_model.pkl"
	with open(model_load_path,'rb') as file:
		unpickled_model = pickle.load(file)

	# Predicting the grade with the model
	prediction = unpickled_model.predict(preproc)

	# Changing the prediction to a 100 point scale so South Africans can understand
	prediction = prediction * 5

	# Heading
	st.title("The predicted grade is :")
	# Write the predicted grade
	st.write(prediction[0])

	# Different output based on if the student is predicted to fail 
	if prediction >= 50:	
		st.image("graduation-hat.png")
		st.write("This student is likely to pass")
	elif prediction >= 40:
		st.image("person.png")
		st.write("This student is at risk of failing")
	else:
		st.image("caution.png")
		st.write("This student is at high risk of failing")

# Run the app		
if __name__ == '__main__':
	main()