#!/bin/bash

# The purpose of this script is to 1) unzip the json.gz files; 2) convert to json; and
# 3) output to a renamed file that is the projectmemberid's filetype.csv

# This was designed based on the Nightscout data source type from the Nightscout Data Trasnfer app
# This pulls profile, entries, treatments, and devicestatus data
# You can easily sub in different data $type file names in the future;
# the first for loop is for special purpose, but second for loop is the most general purpose.

#run from the folder where your OH data is downloaded to

ls -d [0-9]* | while read dir; do

    # Print the directory/folder name you're investigating
    echo $dir

    if [ -d "$dir/direct-sharing-31/" ]; then

        cd $dir/direct-sharing-31/

        # if date run has a space in .gz file name, remove the space before processing
        for f in *\ *; do mv "$f" "${f// /}"; done &>/dev/null
        
        # exit the script right away if something fails
        # set -eu

        #unzip the relevant json file and re-name it with the directory name as a json
        type=treatments
        ls ${type}*.gz | while read file; do    

            gzip -d ${file} || echo "${dir}_${file} is not gzip"
            #gzip -cd entries.json.gz > ${dir}_entries.json

            # print the name of the new file, to confirm it unzipped successfully
            echo "${file} done"

     
            # pipe the json into csv, taking the dateString and sgv datums
            # if cat ${dir}_${file} | jq -e .[0] > /dev/null; then
            #   echo "${dir}_${file} is good to go"
            # else
            #   echo "${dir}_${file} does not appear to be valid json, or is empty"
            # fi
           
        done
        cd ../../

    fi


    # print copy done, so you know that it made it through a full cycle on a single data folder
    #echo "Copy done"

done


       
