dir_data_local_fp=/cygdrive/c/Raymetrics/Lidar_Data/Fixed_Point/
dir_data_remoto_fp=/mnt/ALMACENAMIENTO/LIDAR/LastFiles/Fixed_Point/
dir_data_local_sm=/cygdrive/c/Raymetrics/Lidar_Data/Scanning_Measurements/
dir_data_remoto_sm=/mnt/ALMACENAMIENTO/LIDAR/Scanning_Measurements/

echo $dir_data_local_fp
echo $dir_data_remoto_fp
echo $dir_data_local_sm
echo $dir_data_remoto_sm

last_fp=`ls -t1 $dir_data_local_fp/ | head -n1`
last_sm=`ls -t1 $dir_data_local_sm/ | head -n1`

echo $last_fp
echo $last_sm

#scp -r $dir_data_local_fp$last_fp/* torresiata@192.168.1.62:$dir_data_remoto_fp
#scp -r $dir_data_local_sm$last_sm torresiata@192.168.1.62:$dir_data_remoto_sm

rsync -razP $dir_data_local_fp$last_fp/* torresiata@192.168.1.62:$dir_data_remoto_fp
rsync -razP $dir_data_local_sm$last_sm torresiata@192.168.1.62:$dir_data_remoto_sm

ssh torresiata@192.168.1.62 "chmod -R ugo+rx $dir_data_remoto_fp"
ssh torresiata@192.168.1.62 "chmod -R ugo+rx $dir_data_remoto_sm"
