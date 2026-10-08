#!/bin/bash

INDIR=$1
OUTDIR=$2
LISTFILE=$3

OUTPUTROOT=$OUTDIR
TEMPDIR=/temp

# per-case preprocessing cache (bin/prepcache.py); set SARCO_CACHE_DIR to keep
# it across runs, e.g. on a persistent volume
CACHEDIR=${SARCO_CACHE_DIR:-$TEMPDIR/cache}

mkdir -p $TEMPDIR

mkdir -p $OUTPUTROOT

WORKDIRROOT=$TEMPDIR/work
RESULTDIR=$OUTPUTROOT

mkdir -p $WORKDIRROOT
mkdir -p $RESULTDIR

WORKDIR_L1=$WORKDIRROOT/l1
WORKDIR_L3=$WORKDIRROOT/l3

mkdir -p $WORKDIR_L1
mkdir -p $WORKDIR_L3

TEMPLISTFILE=$WORKDIRROOT/list.txt





python bin/loadfiles.py $LISTFILE $TEMPLISTFILE $INDIR $WORKDIRROOT

python bin/lumbar.py --mode test --volume_list $TEMPLISTFILE --model_file_name model/l1_lumbar.ckpt --output_file_name $WORKDIR_L1/l1.txt --labelused 1 --device cuda --cache_dir $CACHEDIR

python bin/lumbar.py --mode test --volume_list $TEMPLISTFILE --model_file_name model/l3_lumbar.ckpt --output_file_name $WORKDIR_L3/l3.txt --labelused 2 --device cuda --cache_dir $CACHEDIR

python bin/lumbarmodifier.py --mode passthrough --volume_list $TEMPLISTFILE --input_file_name=$WORKDIR_L1/l1.txt --output_file_name=$WORKDIR_L1/l1mod.txt

python bin/lumbarmodifier.py --mode passthrough --volume_list $TEMPLISTFILE --input_file_name=$WORKDIR_L3/l3.txt --output_file_name=$WORKDIR_L3/l3mod.txt

python bin/sarcosegmentation.py --mode test --volume_list_file=$WORKDIR_L1/l1mod.txt --z_list_file=$WORKDIR_L1/l1mod.txt --device cuda --workdir=$WORKDIR_L1 --model_file_name=model/l1_segmentation.ckpt --logfilepath=$WORKDIR_L1/log.txt --resultdir=$RESULTDIR --desc=L1 --cache_dir $CACHEDIR

python bin/sarcosegmentation.py --mode test --volume_list_file=$WORKDIR_L3/l3mod.txt --z_list_file=$WORKDIR_L3/l3mod.txt --device cuda --workdir=$WORKDIR_L3 --model_file_name=model/l3_segmentation.ckpt --logfilepath=$WORKDIR_L3/log.txt --resultdir=$RESULTDIR --desc=L3 --cache_dir $CACHEDIR

python bin/resultuniter_circus.py $RESULTDIR/__result_L1.csv $RESULTDIR/__result_L3.csv $RESULTDIR/results.json





