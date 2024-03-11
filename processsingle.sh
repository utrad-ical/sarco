#!/bin/bash

INDIR=$1
OUTDIR=$2

OUTPUTROOT=$OUTDIR

LISTFILE=$OUTDIR/list.txt

> $LISTFILE echo {volume_name}
>> $LISTFILE echo $INDIR/0.mhd

mkdir -p $OUTPUTROOT

WORKDIRROOT=$OUTPUTROOT/work
RESULTDIR=$OUTPUTROOT/result

mkdir -p $WORKDIRROOT
mkdir -p $RESULTDIR

WORKDIR_L1=$WORKDIRROOT/l1
WORKDIR_L3=$WORKDIRROOT/l3

mkdir -p $WORKDIR_L1
mkdir -p $WORKDIR_L3







python bin/lumbar.py --mode test --volume_list $LISTFILE --model_file_name model/l1_lumbar.ckpt --output_file_name $WORKDIR_L1/l1.txt --labelused 1 --device cuda

python bin/lumbar.py --mode test --volume_list $LISTFILE --model_file_name model/l3_lumbar.ckpt --output_file_name $WORKDIR_L3/l3.txt --labelused 2 --device cuda

python bin/lumbarmodifier.py --mode test --volume_list $LISTFILE --device=cuda --model_file_name=model/l1_modifier.ckpt --input_file_name=$WORKDIR_L1/l1.txt --output_file_name=$WORKDIR_L1/l1mod.txt --labelused 1

python bin/lumbarmodifier.py --mode test --volume_list $LISTFILE --device=cuda --model_file_name=model/l3_modifier.ckpt --input_file_name=$WORKDIR_L3/l3.txt --output_file_name=$WORKDIR_L3/l3mod.txt --labelused 2

python bin/sarcosegmentation.py --mode test --volume_list_file=$WORKDIR_L1/l1mod.txt --z_list_file=$WORKDIR_L1/l1mod.txt --device cuda --workdir=$WORKDIR_L1 --model_file_name=model/l1_segmentation.ckpt --logfilepath=$WORKDIR_L1/log.txt --resultdir=$RESULTDIR --desc=L1

python bin/sarcosegmentation.py --mode test --volume_list_file=$WORKDIR_L3/l3mod.txt --z_list_file=$WORKDIR_L3/l3mod.txt --device cuda --workdir=$WORKDIR_L3 --model_file_name=model/l3_segmentation.ckpt --logfilepath=$WORKDIR_L3/log.txt --resultdir=$RESULTDIR --desc=L3

python bin/resultuniter.py $RESULTDIR/__result_L1.csv $RESULTDIR/__result_L3.csv $RESULTDIR/__result_united.csv





