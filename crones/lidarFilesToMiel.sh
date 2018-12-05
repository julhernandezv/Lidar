#!/bin/bash

check_folder () {
    printf "user@domain = $1 \npath = $2\n"
    if ssh $1 "test -d $2"; then
        printf "That directory exists\n"
    else
        printf "That directory doesn't exists, creating... \n"
        ssh $1 "mkdir -p -m 755 '$2'" #2> stderr
        # cat
    fi
}

rsync_wrapper () {
    # printf "Args number: $#\n"
    local POSITIONAL=()

    while [[ $# -gt 0 ]]; do
    local key="$1"

    case $key in
        -lp|--local-path)
        local SOURCE="$2"; shift; shift;; # past argument, # past value
        -rp|--path)
        local REMOTEPATH="$2"; shift; shift;;
        -f|--files-from)
        local FILES="$2"; shift; shift;;
        -t|--type)
        local TYPE="$2"; shift; shift;;
        *)    # unknown option
        POSITIONAL+=("$1")  shift ;; #save it in an array for later
    esac
    done
    set -- "${POSITIONAL[@]}" # restore positional parameters
    # local SOURCE="$1"
    # local REMOTEPATH="$2"
    # local TYPE="$3"
    # local FILES="$4"

    if [ "$TYPE" = "d" ]; then
        printf "Rsync folders \nsource=$SOURCE \npath=$REMOTEPATH \ninclude-from=$FILES\n"
        rsync -tzvP --chmod=755 --include-from="$FILES" "$SOURCE" "$REMOTEPATH" 
    elif [ "$TYPE" = "f" ]; then
        printf "Rsync files \nsource=$SOURCE \npath=$PATH \nfiles-from=$FILES\n"
        rsync -tzvP --chmod=755 --files-from="$FILES" --no-relative --no-dirs "$SOURCE" "$REMOTEPATH" #
    else
        printf "Error on parameters, check it: $@\n"
    fi
}

verify_ssh_connection () {
    printf "\nFunction: $@\n"
    for ((i=1;i<=10;i++))
    do
        printf "\nCount: $i\n"
        local out=$("$@" 2>$1)
        printf "Out:\n%s" "$out"
        printf "\nError: $?\n"

        if [ $? -eq 0 ]; then
            printf "%s\n" "SSH command successful"
            break
        elif [ $? -ne 255 ]; then
            printf "%s\n%s\n" "SSH command failed" "$out"
            break
        else
            printf "%s\n%s\n" "SSH connection refused" "$out"
            sleep 2
        fi
    done
}


rsync_between_dates () {

    local POSITIONAL=()
    local START=$(date -d "30 min ago" '+%Y%m%d %H%M')
    local END=$(date '+%Y%m%d %H%M')
    local TYPE="f"
    local NAME="*"
    local MINDEPTH=2

    while [[ $# -gt 0 ]]; do
    local key="$1"

    case $key in
        -lp|--local-path)
        local LOCALPATH="$2"; shift; shift;; # past argument, # past value
        -rp|--remote-path)
        local REMOTEPATH="$2"; shift; shift;;
        -d|--domain)
        local DOMAIN="$2"; shift; shift;;
        -s|--start)
        START="$2"; shift; shift;;
        -e|--end)
        END="$2"; shift; shift;;
        -t|--type)
        TYPE="$2"; shift; shift;;
        -n|--name)
        NAME="$2"; shift; shift;;
        -mindepth)
        MINDEPTH="$2"; shift; shift;;
        *)    # unknown option
        POSITIONAL+=("$1")  shift ;; #save it in an array for later
    esac
    done
    set -- "${POSITIONAL[@]}" # restore positional parameters

    local date_folder=$(date "-d $START + 9 min" '+%Y%m%d' )
    if [ "$TYPE" = "d" ]; then
        date_folder="$date_folder/"
    fi
    printf "start: %s\nend: %s\nlocal path: %s\nremote path: %s\nuser@domain: %s\ntype: %s\nname: %s\n"  "$START" "$END" "$LOCALPATH" "$REMOTEPATH$date_folder" "$DOMAIN" "$TYPE" "$NAME";
    find "$LOCALPATH" -mindepth $MINDEPTH -name "$NAME" -newermt "$START" ! -newermt "$END" -type "$TYPE" -printf '%P\n'  > files.txt

    files=$(cat files.txt)
    # printf "$files"
    if [ -n "$files" ]; then
        verify_ssh_connection check_folder "$DOMAIN" "$REMOTEPATH$date_folder"
        verify_ssh_connection rsync_wrapper -lp "$LOCALPATH" -rp "$DOMAIN:$REMOTEPATH$date_folder" -t "$TYPE" -f files.txt
    else
        printf "\nNo files avalible, check other dates\n"
    fi
}
########################################

miel=torresiata@192.168.1.62
gomita=jhernandezv@192.168.1.9

dir_data_local_fp=/cygdrive/c/Raymetrics/Lidar_Data/Fixed_Point/
dir_data_local_sm=/cygdrive/c/Raymetrics/Lidar_Data/Scanning_Measurements/
dir_miel_fp=/mnt/ALMACENAMIENTO/LIDAR/Fixed_Point/
dir_miel_sm=/mnt/ALMACENAMIENTO/LIDAR/Scanning_Measurements/
dir_gomita_fp=/media/jhernandezv/disco1/Lidar/Fixed_Point/
dir_gomita_sm=/media/jhernandezv/disco1/Lidar/Scanning_Measurements/

########################################

# if [ $(date '+%Y%m%d') -gt $(date -d "30 min ago" '+%Y%m%d') ]; then
#     start1=$(date -d "30 min ago" '+%Y%m%d 2330')
#     end1=$(date '-d 9 min ago' '+%Y%m%d 0000')
#     start2=$end1
#     end2=$(date '+%Y%m%d 0030')
#     printf "Sync on day change\n"
#     #FixedPoint
#     rsync_between_dates -lp $dir_data_local_fp -rp $dir_miel_fp -d $miel -n "RM*" -s "$start1" -e "$end1"
#     rsync_between_dates -lp $dir_data_local_fp -rp $dir_gomita_fp -d $gomita -n "RM*" -s "$start1" -e "$end1"
#
#     rsync_between_dates -lp $dir_data_local_fp -rp $dir_miel_fp -d $miel -n "RM*" -s "$start2" -e "$end2"
#     rsync_between_dates -lp $dir_data_local_fp -rp $dir_gomita_fp -d $gomita -n "RM*" -s "$start2" -e "$end2"
#
#     #Scanning
#     rsync_between_dates -lp $dir_data_local_sm -rp $dir_miel_sm -d $miel -t d -s "$start1" -e "$end1"
#     rsync_between_dates -lp $dir_data_local_sm -rp $dir_gomita_sm -d $gomita -t d -s "$start1" -e "$end1"
#
#     rsync_between_dates -lp $dir_data_local_sm -rp $dir_miel_sm -d $miel -t d -s "$start2" -e "$end2"
#     rsync_between_dates -lp $dir_data_local_sm -rp $dir_gomita_sm -d $gomita -t d -s "$start2" -e "$end2"
# else
#     printf "Sync Files\n"
#     #FixedPoint
#     rsync_between_dates -lp $dir_data_local_fp -rp $dir_miel_fp -d $miel -n "RM*"
#     rsync_between_dates -lp $dir_data_local_fp -rp $dir_gomita_fp -d $gomita -n "RM*"
#
#     #Scanning
#     rsync_between_dates -lp $dir_data_local_sm -rp $dir_miel_sm -d $miel -t d
#     rsync_between_dates -lp $dir_data_local_sm -rp $dir_gomita_sm -d $gomita -t d
# fi
#

###############################################################################
# Rsync daily
# today="20181204"
# rsync_wrapper "$dir_data_local_fp" "$gomita:$dir_gomita_fp$today" f files.txt
# echo "Direct:\n"
# rsync -tzvP --chmod=755 --no-relative --no-dirs --files-from=files.txt "$dir_data_local_fp" "$gomita:$dir_gomita_fp$today"

today=$(date '+%Y%m%d %H%M')
today="20181129"
# d="20180201 0000"
d="20181128 0000"
d=$(date "-d $d - 9 min" '+%Y%m%d %H%M')
# today="20181107"
until [[ "$d" > "$today" ]];
do

    df=$(date -d "$d + 1 day" '+%Y%m%d %H%M')
    printf "\nSync files from $d to $df\n \n"
    #FixedPoint
    # rsync_between_dates -lp $dir_data_local_fp -s "$d" -e "$df" -rp $dir_miel_fp -d$miel -t f -n "RM*"
    # rsync_between_dates -lp $dir_data_local_fp -s "$d" -e "$df" -rp $dir_gomita_fp -d $gomita -t f -n "RM*"

    # Scanning
    echo $(find $dir_data_local_sm -mindepth 2 -name "*" -newermt "$d" ! -newermt "$df" -type d -printf '%P\n')
    # rsync_between_dates -lp $dir_data_local_sm -s "$d" -e "$df" -rp $dir_miel_sm -d $miel -t d
    rsync_between_dates -lp $dir_data_local_sm -s "$d" -e "$df" -rp $dir_gomita_sm -d $gomita -t d

    d=$df
done
#

###############################################################################
# #
#
#
# last_fp=`ls -t1 $dir_data_local_fp | head -n1`
# last_sm=`ls -t1 $dir_data_local_sm | head -n1`
# # last_folder_sm=`ls -t1 $dir_data_local_sm$last_sm/ | head -n1`
#
# echo $last_fp
# echo $last_sm
# # echo $last_folder_sm
#
# #----------------------------------
# #FixedPoint
#
# files_fp=`find $dir_data_local_fp$last_fp/ -name 'RM*' -mmin -15` # > /home/siata/lastFilesFP.txt
# echo "Latest Fixed Point Files$\n: $files_fp"
#
# if [ -n "$files_fp" ]; then
#
#
#     verify_ssh_connection check_folder $server $dir_miel_fp
#     # ssh $server "mkdir -p '$dir_miel_fp'"
#     find $dir_data_local_fp$last_fp/ -name 'RM*' -mmin -15 -type f -printf "%f\n" > /home/siata/lastFilesFP.txt
#     verify_ssh_connection rsync_wrapper $dir_data_local_fp$last_fp/ $server:$dir_miel_fp lastFilesFP.txt
#     # rsync -trzvP --chmod=ugo+rx --files-from=lastFilesFP.txt $dir_data_local_fp$last_fp/ $server:$dir_miel_fp
#
# else
#     echo "No Fixed Point files avalible"
# fi
#
# #----------------------------------
# #Scanning
#
# # files_sm=`find $dir_data_local_sm$last_sm/$last_folder_sm/ -name 'RM*' -mmin -5`
# # latest_files_sm=`find $dir_data_local_sm$last_sm/ -name '*' -type f -printf "%f\n"`
# latest_folders_sm=`find $dir_data_local_sm$last_sm/ -name '*' -type d -mmin 15`
# echo "Latest Scanning Measurements Folders: $latest_folders_sm"
# # echo "Latest Scanning Measurements Folders: $latest_folders_sm"
# #`find $dir_data_local_sm$last_sm/ -name '*' -mmin -5`  #> /home/siata/lastFilesSM.txt
# # mapfile -t latest < <( `find $dir_data_local_sm$last_sm/ -name '*' -type d` )
# # for fol in ${latest_folders_sm:2}
# # do
# #     echo fol
# # done
#
# if [ -n "$latest_folders_sm" ]; then
#     for folder in $latest_folders_sm
#     do
#         echo "Folder: $folder"
#         directory=${folder/$dir_data_local_sm$last_sm}
#         # echo "Directory: $directory"
#         # echo "Path: $dir_data_local_sm$last_sm"
#         if [ $folder != $dir_data_local_sm$last_sm/ ]; then
#             verify_ssh_connection check_folder $server $dir_miel_sm$last_sm$directory
#             # ssh $server "mkdir -p '$dir_miel_sm$last_sm/$folder/'"
#             verify_ssh_connection rsync_wrapper $dir_data_local_sm$last_sm$directory/ $server:$dir_miel_sm$last_sm$directory
#         fi
#
#     done
#     echo All done
# else
#     echo "No Scanning Measurements files avalible"
# fi
#
# #scp -r $dir_data_local_fp$last_fp/* $server:$dir_miel_fp
# #scp -r $dir_data_local_sm$last_sm $server:$dir_miel_sm
#
# #rsync -trzP --files-from=/home/siata/lastFilesFP.txt $server:$dir_miel_fp
# #rsync -trzP --files-from=/home/siata/lastFilesSM.txt $server:$dir_miel_sm
#
#
# #ssh $server "chmod -R ugo+rx $dir_miel_sm"
