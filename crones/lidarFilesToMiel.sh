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
    if [ "$#" -eq 2 ]; then
        printf "Rsync \nsource=$1 \npath=$2\n"
        rsync -trmzvP --chmod=755 "$1" "$2"
    elif [ "$#" -eq 3 ]; then
        printf "Rsync \nsource=$1 \npath=$2 \nfiles-from=$3\n"
        rsync -tmzvP --chmod=755 --no-relative --no-dirs --delete --files-from="$3" "$1" "$2"
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
    local END=$(date '+%Y%m%d %H%M')
    local START=$(date -d "$END - 30 min" '+%Y%m%d %H%M')
    local TYPE="f"
    local NAME="*"

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
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift ;;
    esac
    done
    set -- "${POSITIONAL[@]}" # restore positional parameters

    printf "local path: %s\nstart: %s\nend: %s\nremote path: %s\nuser@domain: %s\ntype: %s\nname: %s\n" "$LOCALPATH" "$START" "$END" "$REMOTEPATH" "$DOMAIN" "$TYPE" "$NAME";
    find $LOCALPATH -name $NAME -newermt "$START" ! -newermt "$END" -type $TYPE -printf '%P\n' > files.txt
    files=$(cat files.txt)
    # printf "$files"
    local date_folder=$(date "-d $START" '+%Y%m%d' )
    if [ -n "$files" ]; then
        verify_ssh_connection check_folder $DOMAIN $REMOTEPATH$date_folder
        verify_ssh_connection rsync_wrapper $LOCALPATH $DOMAIN:$REMOTEPATH$date_folder files.txt
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

# echo $dir_data_local_fp
# echo $dir_miel_fp
# echo $dir_data_local_sm
# echo $dir_miel_sm
#`date -d '5 hour ago' '+%Y-%m-%d -%H:%M'`
# if [ "`date -d '+5 hour' '+%H:%M'`" = '00:00' ]; then
#     today=`date -d '+4 hour' '+%Y%m%d'`
#     echo "Files to previous day"
# else
#     today=`date -d '+5 hour' '+%Y%m%d'`
# fi


rsync_between_dates -lp $dir_data_local_fp -rp $dir_miel_fp -d $miel -n "RM*"
rsync_between_dates -lp $dir_data_local_fp -rp $dir_gomita_fp -d $gomita -n "RM*"

#Scanning
rsync_between_dates -lp $dir_data_local_sm -rp $dir_miel_sm -d $miel -t d
rsync_between_dates -lp $dir_data_local_sm -rp $dir_gomita_sm -d $gomita -t d
#
# today=$(date '+%Y%m%d')
# d="2018-02-01"
# d=$(date "-d $d - 9 min" '+%Y%m%d %H%M')
# # today="20181107"
# # d="20181107"
# until [[ "$d" > "$today" ]];
# do
#
#     df=$(date -d "$d + 1 day" '+%Y%m%d %H%M')
#     printf "\nSync files from $d to $df\n \n"
#     #FixedPoint
#     rsync_between_dates -lp $dir_data_local_fp -s $d -e $df -rp $dir_miel_fp -d$miel -t f -n "RM*"
#     rsync_between_dates -lp $dir_data_local_fp -s $d -e $df -rp $dir_gomita_fp -d $gomita -t f -n "RM*"
#
#     #Scanning
#     rsync_between_dates -lp $dir_data_local_sm -s $d -e $df -rp $dir_miel_sm -d $miel -t d
#     rsync_between_dates -lp $dir_data_local_sm -s $d -e $df -rp $dir_gomita_sm -d $gomita -t d
#
#     d=$df
# done

#
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
