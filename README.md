# AntiSem2020
This is the github repo of Group 2 of the Datathon and Hackathon organized by the INSTITUTE FOR THE STUDY OF CONTEMPORARY ANTISEMITISM - Indiana University.

## How to run our model with a given json
Make sure you run python3 and you installed all the libs

To feed our model with tweets and receive a csv file (tweet id, evaluation) simply run
```
python evaluate_tweets.py
```
The program will ask you to enter the file/path name of the given json
```
Enter the path/name of the json file to read
```
The evaluation data is allready in the directory **input_data**

Enter
```
input_data/to_test.json
```

When the script is done you can find a csv with an evaluation in the directory **results**