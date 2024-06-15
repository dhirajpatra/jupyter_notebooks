# Open Weather Challenge
SparkCognition Renewables Suite provides insights and alerts to our customers for Wind and Solar Assets. Weather is a key component of some of these alerts and your goal with the challenge is to create a service that can send daily weather forecast for some location via an email

To achieve this use Weather API from OpenWeather (https://openweathermap.org/) to get the latest forecast information for given lat,long and send that as an email. Please document your design and share implementation to run the same.

## Mandatory Constraints/Requirements
1. Use Python for weather and email scripts/code
2. Use AWS for any other cloud services or architecture design recommendations

## General Guidance/Steps:
1. Sign up for Open Weather to get weather forecast for your local city
2. Setup Anaconda and Python Virtual Environment/Jupyter Notebook Setup
3. Python Notebook 1: Visualize Weather Information from Open Weather Map using visualization library such as Matplotlib, Seaborn, Plotly
4. Python Notebook 2: Write Notebook to send emails by specifying a to email
    Document steps to run the same and reason for your choice of method and if you will make any changes or design choices when moving this to production
5. Send Weather Information Using Email
   i.  You now should have 2 notebooks: One to visualize weather data and another to send email
   ii. In this step create a new notebook that combines the 2 notebooks to get the current and forecast weather information, formats it nicely in HTML and sends an email to you with Weather Information when notebook is run
6. Suggest automation using serverless solution such as AWS Lambda to send daily weather forecast data via email

## Deliverable
Submit the 3 notebooks with following names:
Python Notebook 1: notebook_1_your_name.ipynb -- Notebook that can take lat long inputs and generate weather report/forecast with charts for the location
Python Notebook 2: notebook_2_your_name.ipynb -- Notebook that has function to send an email to given email_id
Python Notebook 3: notebook_3_your_name.ipynb -- Notebook that can take lat/long, email_id as input at top and when run sends an email of weather report

## Bonus Deliverable
If you complete step 6 of automation then share the implementation details, design choices and deployable code with steps to deploy in our AWS account
