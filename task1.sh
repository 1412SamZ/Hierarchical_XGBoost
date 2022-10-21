#!/bin/bash
#SBATCH --job-name=HXGB_T1  # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --mail-user=zhanggua@tnt.uni-hannover.de   # only <UserName>@tnt.uni-hannover.de is allowed as mail address
#SBATCH --mail-type=ALL  # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --output=tmp/slurm_logs/task1-%j.txt  # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)
#SBATCH --time=1-0                     # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS oder Tage-Stunden)
#SBATCH --partition=cpu_normal_stud    # Partition auf der gerechnet werden soll (Bei GPU Jobs unbedingt notwendig)
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G

                                    

echo "Hier beginnt die Ausführung/Berechnung"
working_dir=~
env_dir=~/anaconda3/envs/hxgboost
project_dir=~/HXGBoost
cp ~/withlock $env_dir/bin

cd $working_dir
srun hostname

# Kopiere Trainingsdaten auf den Rechennode
# Im eigentlichen Job kann der Speicherort der Trainingsdaten aus der Umgebungsvariable SLURM_TRAINING_DATA_PATH gelesen werden.
DATA_SOURCE=/home/zhanggua/HXGBoost/dataset/
DATA_TARGET=tmp/slurm/
srun bash -c "[ -d "/localstorage/${USER}/${DATA_TARGET}" ] || mkdir -p "/localstorage/${USER}/${DATA_TARGET}""
# Withlock ist ein Python Skript, das auf den Nodes installiert sein sollte. Damit lassen sich Mutexe erzeugen.
# Withlock überprüft, ob die Datei sync.lock existiert, falls nicht wird die Datei erstellt und rsync ausgeführt. 
# Falls sync.lock existiert, wartet withlock bis die Datei nicht mehr existiert und führt dann rsync aus.
# Das wird gemacht, um zu verhindern das rsync auf dem selben Node mehrmals gleichzeitig ausgeführt wird.
srun python2 $env_dir/bin/withlock -w 86400 "/localstorage/${USER}/sync.lock" rsync -avHAPSx "$DATA_SOURCE" "/localstorage/${USER}/${DATA_TARGET}"
export SLURM_TRAINING_DATA_PATH="/localstorage/${USER}/${DATA_TARGET}"

# activate virtualenv
source /home/zhanggua/anaconda3/bin/activate
conda activate hxgboost
# run code 
cd $project_dir

srun python3 training.py $SLURM_TRAINING_DATA_PATH