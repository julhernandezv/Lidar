#!/bin/bash

check_folder () {
    echo "Arg 1: server@server = $1"
    echo "Arg 2: path = $2"
    if ssh $1 "test -d $2"; then
        echo "That directory exists"
    else
        echo "That directory doesn't exists, creating..."
        ssh $1 "mkdir -p -m 755 '$2'" #2> stderr
        # cat
    fi
}

rsync_wrapper () {
    echo "$#"
    if [ "$#" -eq 2 ]; then
        echo "Rsync source=$1 path=$2"
        rsync -trzvP --chmod=755 "$1" "$2"
    elif [ "$#" -eq 3 ]; then
        echo "Rsync source=$1 path=$2 files-from=$3"
        rsync -trzvP --chmod=755 --files-from="$3" "$1" "$2"
    else
        echo "Error on parameters, check it: $@"
    fi
}

verify_ssh_connection () {
    echo "$@"
    local out=$("$@" 2>$1)
    echo $out
    echo $?
    if [ $? -eq 0 ]; then
        printf "%s\n" "SSH command successful"
    elif [ $? -eq 255 ]; then

        # while [ $? -eq 255 ]
        for ((i=1;i<=10;i++))
        do
            printf "%s\n%s" "SSH failed with following error:" "$out"
            sleep 5
            local out=$("$@" 2>$1)
            if [ $? -eq 0 ]; then
                printf "%s\n" "SSH command successful"
                break
            fi
        done
    else
        printf "%s\n%s" "ls command failed" "$out"
    fi
}


#`date -d '5 hour ago' '+%Y-%m-%d -%H:%M'`
if [ "`date -d '+5 hour' '+%H:%M'`" = '00:00' ]; then
    today=`date -d '+4 hour' '+%Y%m%d'`
    echo "Files to previous day"
else
    today=`date -d '+5 hour' '+%Y%m%d'`
fi
# today='20181106'
server=torresiata@192.168.1.62
# test1=/home/torresiata/test/
# test2=/home/torresiata/test2/
# echo Test check folder
# check_folder $server $test1
# check_folder $server $test2

dir_data_local_fp=/cygdrive/c/Raymetrics/Lidar_Data/Fixed_Point/
dir_data_local_sm=/cygdrive/c/Raymetrics/Lidar_Data/Scanning_Measurements/
dir_data_remoto_fp=/mnt/ALMACENAMIENTO/LIDAR/Fixed_Point/$today
dir_data_remoto_sm=/mnt/ALMACENAMIENTO/LIDAR/Scanning_Measurements/

echo $dir_data_local_fp
echo $dir_data_remoto_fp
echo $dir_data_local_sm
echo $dir_data_remoto_sm

last_fp=`ls -t1 $dir_data_local_fp | head -n1`
last_sm=`ls -t1 $dir_data_local_sm | head -n1`
# last_folder_sm=`ls -t1 $dir_data_local_sm$last_sm/ | head -n1`

echo $last_fp
echo $last_sm
# echo $last_folder_sm

#----------------------------------
#FixedPoint

files_fp=`find $dir_data_local_fp$last_fp/ -name 'RM*' -mmin -15` # > /home/siata/lastFilesFP.txt
echo "Latest Fixed Point Files: $files_fp"

if [ -n "$files_fp" ]; then


    verify_ssh_connection check_folder $server $dir_data_remoto_fp
    # ssh $server "mkdir -p '$dir_data_remoto_fp'"
    find $dir_data_local_fp$last_fp/ -name 'RM*' -mmin -15 -type f -printf "%f\n" > /home/siata/lastFilesFP.txt
    verify_ssh_connection rsync_wrapper $dir_data_local_fp$last_fp/ $server:$dir_data_remoto_fp lastFilesFP.txt
    # rsync -trzvP --chmod=ugo+rx --files-from=lastFilesFP.txt $dir_data_local_fp$last_fp/ $server:$dir_data_remoto_fp

else
    echo "No Fixed Point files avalible"
fi

#----------------------------------
#Scanning

# files_sm=`find $dir_data_local_sm$last_sm/$last_folder_sm/ -name 'RM*' -mmin -5`
# latest_files_sm=`find $dir_data_local_sm$last_sm/ -name '*' -type f -printf "%f\n"`
latest_folders_sm=`find $dir_data_local_sm$last_sm/ -name '*' -type d -mmin 15`
echo "Latest Scanning Measurements Folders: $latest_folders_sm"
# echo "Latest Scanning Measurements Folders: $latest_folders_sm"
#`find $dir_data_local_sm$last_sm/ -name '*' -mmin -5`  #> /home/siata/lastFilesSM.txt
# mapfile -t latest < <( `find $dir_data_local_sm$last_sm/ -name '*' -type d` )
# for fol in ${latest_folders_sm:2}
# do
#     echo fol
# done

if [ -n "$latest_folders_sm" ]; then
    for folder in $latest_folders_sm
    do
        echo "Folder: $folder"
        directory=${folder/$dir_data_local_sm$last_sm}
        # echo "Directory: $directory"
        # echo "Path: $dir_data_local_sm$last_sm"
        if [ $folder != $dir_data_local_sm$last_sm/ ]; then
            verify_ssh_connection check_folder $server $dir_data_remoto_sm$last_sm$directory
            # ssh $server "mkdir -p '$dir_data_remoto_sm$last_sm/$folder/'"
            verify_ssh_connection rsync_wrapper $dir_data_local_sm$last_sm$directory/ $server:$dir_data_remoto_sm$last_sm$directory
        fi

    done
    echo All done
else
    echo "No Scanning Measurements files avalible"
fi

#scp -r $dir_data_local_fp$last_fp/* $server:$dir_data_remoto_fp
#scp -r $dir_data_local_sm$last_sm $server:$dir_data_remoto_sm

#rsync -trzP --files-from=/home/siata/lastFilesFP.txt $server:$dir_data_remoto_fp
#rsync -trzP --files-from=/home/siata/lastFilesSM.txt $server:$dir_data_remoto_sm


#ssh $server "chmod -R ugo+rx $dir_data_remoto_sm"
