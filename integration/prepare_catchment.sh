# usage: ./prepare_catchment.sh <path to calibration workflow root dir> <NUM_GPUA> <catchID> <path to lisflood-calibration sources root>
set -euo pipefail
root_dir=$1
NCPUS=$2
OBSID=$3
src_root=$4

catchment_root=$root_dir/catchments/$OBSID
templates_dir=$root_dir/templates
stations_dir=$root_dir/stations

cp $templates_dir/settings.txt $catchment_root/settings.txt
sed -i "s:STATIONS:$stations_dir:" $catchment_root/settings.txt
sed -i "s:TEMPLATES:$templates_dir:" $catchment_root/settings.txt
sed -i "s:OBS:$stations_dir/observed_discharges.csv:" $catchment_root/settings.txt
sed -i "s:NCPUS:$NCPUS:" $catchment_root/settings.txt
sed -i "s:CATCHMENTS_DIR:$root_dir/catchments:" $catchment_root/settings.txt

#time python $src_root/bin/CAL_4_EXTRACT_STATION.py $catchment_root/settings.txt $OBSID

