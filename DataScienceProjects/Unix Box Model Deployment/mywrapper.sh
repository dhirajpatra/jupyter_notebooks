#!/bin/bash
#Create Log file
touch /home/hdoop/ModelDeployment/Logs/Script_logs.log

LOG_FILE=/home/hdoop/ModelDeployment/Logs/Script_logs.log

echo Script execution started at: `date +"%Y-%m-%d %T"` >>  $LOG_FILE

#Checking Number of Prameters Passed
if [ $# -ne 1 ]
then		
		echo Invalid number of parameters passed
		exit 1
fi
echo Parameter check completed >>  $LOG_FILE

#Checking If the directory exist
test_path=$1
if [ -d "$test_path"  ]
then
   echo "Directory exists" >> $LOG_FILE
else
   echo "Directory does not exist, kindly recheck path"
   exit 1
fi

#Checking if Prediction Script Script ran successfully
python3 Scripts/prediction.py $1 
status=$?
[ $status -eq 0 ] && echo "python script ran successfully" || echo "python script failed"
echo Script execution ended at: `date +"%Y-%m-%d %T"` >>  $LOG_FILE