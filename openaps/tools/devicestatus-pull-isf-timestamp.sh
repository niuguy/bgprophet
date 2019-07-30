#The purpose of this script is to 1) pull ISF and timestamp values from devicestatus csv files 2) output to a new csv file.
#This assumes that you have unzipped and .csv devicestatus files. If not, use the script in the OpenHumansDataTools repo prior to running this script.
#install csvkit prior to running the script.

#this script will only run with OpenAPS datasets, not Loop.

#!/bin/bash

#run from the folder where your OH data is downloaded to. 

set -eu

ls -d [0-9]* | while read dir; do

    echo "Creating files for participant" $dir 
   
    cd $dir/direct-sharing-31/${dir}_devicestatus_*csv
   
    mkdir -p ISF_analysis
    
    ls -d *.csv | while read file; do

    #cut timestamp and isf values from the devicestatus files and save to a new csv file
        echo "ISF, PID, Timestamp" > ISF_analysis/${file%.csv}_isf.csv
        paste -d, \
          <(csvcut -c openaps/enacted/reason $file | awk -F "\"*, \"*" '{print $4}' | sed 's/.*://' | awk '{print $1,F}' FS=, OFS=, F="$dir" | sed 1d) \
          <(csvcut -c openaps/enacted/timestamp $file | cut -d, -f1- | sed 1d) \
          >> ISF_analysis/${file%.csv}_isf.csv
        echo ${file%.csv}_isf.csv
    
    done
   
    cd ../../../
 
done
