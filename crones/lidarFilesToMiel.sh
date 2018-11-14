now=`date '+%Y%m%d'`  #`date -d '5 hour ago' '+%Y-%m-%d %H:%M'

dir_data_local_fp=/cygdrive/c/Raymetrics/Lidar_Data/Fixed_Point/
dir_data_remoto_fp=/mnt/ALMACENAMIENTO/LIDAR/ Fixed_Point/$now
dir_data_local_sm=/cygdrive/c/Raymetrics/Lidar_Data/Scanning_Measurements/
dir_data_remoto_sm=/mnt/ALMACENAMIENTO/LIDAR/Scanning_Measurements

echo $dir_data_local_fp
echo $dir_data_remoto_fp
echo $dir_data_local_sm
echo $dir_data_remoto_sm

last_fp=`ls -t1 $dir_data_local_fp | head -n1`
last_sm=`ls -t1 $dir_data_local_sm | head -n1`
last_folder_sm=`ls -t1 $dir_data_local_sm$last_sm/ | head -n1`

echo $last_fp
echo $last_sm
echo $last_folder_sm

files_fp=`find $dir_data_local_fp$last_fp/ -name 'RM*' -mmin -45` # > /home/siata/lastFilesFP.txt
files_sm=`find $dir_data_local_sm$last_sm/$last_folder_sm/ -name 'RM*' -mmin -5`
#`find $dir_data_local_sm$last_sm/ -name '*' -mmin -5`  #> /home/siata/lastFilesSM.txt

echo $files_fp
echo $files_sm


if [ ! -z "$files_fp" ]; then
    find $dir_data_local_fp$last_fp/ -name 'RM*' -mmin -45 > /home/siata/lastFilesFP.txt
    rsync -trzvP --files-from=/home/siata/lastFilesFP.txt torresiata@192.168.1.62:$dir_data_remoto_fp
    # ssh torresiata@192.168.1.62 "chmod -R ugo+rx $dir_data_remoto_fp"
    echo "rsync $files_fp"
else
    echo "No Fixed Point files avalible"
fi

if [ ! -z "$files_sm" ]; then
    files_sm > /home/siata/lastFilesSM.txt
    #rsync -trzvP --files-from=/home/siata/lastFilesSM.txt torresiata@192.168.1.62:$dir_data_remoto_sm/$last_sm/$last_folder_sm
    #ssh torresiata@192.168.1.62 "chmod -R ugo+rx $dir_data_remoto_sm/$last_sm/$last_folder_sm
    echo "rsync $files_sm"
else
    echo "No Scanning Measuremente files avalible"
fi

#scp -r $dir_data_local_fp$last_fp/* torresiata@192.168.1.62:$dir_data_remoto_fp
#scp -r $dir_data_local_sm$last_sm torresiata@192.168.1.62:$dir_data_remoto_sm

#rsync -trzP --files-from=/home/siata/lastFilesFP.txt torresiata@192.168.1.62:$dir_data_remoto_fp
#rsync -trzP --files-from=/home/siata/lastFilesSM.txt torresiata@192.168.1.62:$dir_data_remoto_sm


#ssh torresiata@192.168.1.62 "chmod -R ugo+rx $dir_data_remoto_sm"
