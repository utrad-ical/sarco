#!/bin/bash

INDIR=$1
OUTDIR=$2

OUTPUTROOT=$OUTDIR
TEMPDIR=/temp

# per-case preprocessing cache (bin/prepcache.py); set SARCO_CACHE_DIR to keep
# it across runs, e.g. on a persistent volume
CACHEDIR=${SARCO_CACHE_DIR:-$TEMPDIR/cache}

mkdir -p $TEMPDIR

LISTFILE=$TEMPDIR/list.txt

mkdir -p $OUTPUTROOT

# the pipeline expects the slices head first; a volume that runs feet first is
# reversed here (header kept) and the result labels are put back at the end
# (bin/orientation.py; without model/orientation.pt the input is used as is)
INPUT=$TEMPDIR/input/0.mhd
python bin/orientation.py check --input $INDIR/0.mhd --output $INPUT --report $OUTPUTROOT/__orientation.json --model model/orientation.pt --device cuda

> $LISTFILE echo {volume_name}
>> $LISTFILE echo $INPUT

WORKDIRROOT=$TEMPDIR/work
RESULTDIR=$OUTPUTROOT

mkdir -p $WORKDIRROOT
mkdir -p $RESULTDIR

WORKDIR_L1=$WORKDIRROOT/l1
WORKDIR_L3=$WORKDIRROOT/l3

mkdir -p $WORKDIR_L1
mkdir -p $WORKDIR_L3







python bin/lumbar.py --mode test --volume_list $LISTFILE --model_file_name model/l1_lumbar.ckpt --output_file_name $WORKDIR_L1/l1.txt --labelused 1 --device cuda --cache_dir $CACHEDIR

python bin/lumbar.py --mode test --volume_list $LISTFILE --model_file_name model/l3_lumbar.ckpt --output_file_name $WORKDIR_L3/l3.txt --labelused 2 --device cuda --cache_dir $CACHEDIR

python bin/lumbarmodifier.py --mode passthrough --volume_list $LISTFILE --input_file_name=$WORKDIR_L1/l1.txt --output_file_name=$WORKDIR_L1/l1mod.txt

python bin/lumbarmodifier.py --mode passthrough --volume_list $LISTFILE --input_file_name=$WORKDIR_L3/l3.txt --output_file_name=$WORKDIR_L3/l3mod.txt

python bin/sarcosegmentation.py --mode test --volume_list_file=$WORKDIR_L1/l1mod.txt --z_list_file=$WORKDIR_L1/l1mod.txt --device cuda --workdir=$WORKDIR_L1 --model_file_name=model/l1_segmentation.ckpt --logfilepath=$WORKDIR_L1/log.txt --resultdir=$RESULTDIR --desc=L1 --cache_dir $CACHEDIR

python bin/sarcosegmentation.py --mode test --volume_list_file=$WORKDIR_L3/l3mod.txt --z_list_file=$WORKDIR_L3/l3mod.txt --device cuda --workdir=$WORKDIR_L3 --model_file_name=model/l3_segmentation.ckpt --logfilepath=$WORKDIR_L3/log.txt --resultdir=$RESULTDIR --desc=L3 --cache_dir $CACHEDIR

# age, contrast phase, body size and phase-0 equivalent values (ct25d checkpoints in model/;
# a missing checkpoint only skips that estimate)
python bin/estimate.py --gray $INPUT --resultdir $RESULTDIR --model_dir model --device cuda --cache_dir $CACHEDIR

# result labels and slice numbers back in the input's own slice order
python bin/orientation.py restore --report $RESULTDIR/__orientation.json --resultdir $RESULTDIR

python bin/resultuniter_circus.py $RESULTDIR/__result_L1.csv $RESULTDIR/__result_L3.csv $RESULTDIR/results.json $RESULTDIR/__estimates.csv





