export RLROOT=`pwd`
dataset_dir="gr_datasets"
dataset_path=`pwd`/$dataset_dir
if [ ! -d "$dataset_path" ]; then
    mkdir "$dataset_path"
fi
export DATASETPATH="$dataset_path"
export PYTHONPATH=`pwd`/src:`pwd`

# echo "RLROOT=${RLROOT}" > debug.env
# echo "DATASETPATH=${DATASETPATH}" >> debug.env