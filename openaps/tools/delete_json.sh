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
    # echo $dir
    if [ -d "$dir/direct-sharing-31/" ]; then
	    cd $dir/direct-sharing-31/

	    # exit the script right away if something fails
	    set -eu

	    ls *.json  | while read file; do    

	        echo $file
	        echo 'deleting..'
	        rm $file
	        echo 'done'

	    done
	    cd ../../
	fi

    # print copy done, so you know that it made it through a full cycle on a single data folder
    #echo "Copy done"
done


       
