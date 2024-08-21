#!/bin/sh
echo GOT ARGUEMNT $1
if [ $1 == "fingan" ]
then
    dir=fingan_runs
    glob=Fin
elif [ $1 == "fourierflow" ]
then
    dir=fourierflow_runs
    glob=Four
elif [ $1 == "realnvp" ]
then
    dir=realnvp_runs
    glob=Real
else
    exit 1
fi

echo Write to data/${dir} and glob for ${glob}
mkdir -p  data/${dir}
rsync -azrv -R euler:"/cluster/scratch/grafn/${glob}*/final/*" data/${dir}/
cp -rf data/${dir}/cluster/scratch/grafn/* data/${dir}/
rm -rf data/${dir}/cluster

echo Successful download


