#!/bin/bash

#SBATCH --job-name=transit-access
#SBATCH --account=eubucco
#SBATCH --qos=io
#SBATCH --partition=io
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/log-%j.stdout
#SBATCH --error=logs/log-%j.stderr
#SBATCH --workdir=/p/projects/eubucco/other_projects/transit-access/slurm

mkdir -p logs
module load anaconda
source activate /home/floriann/.conda/envs/transit-access

echo '{"city": "Bogota", "gtfs_path": ["/p/projects/eubucco/other_projects/transit_access/data/gtfs_feeds/bog-GTFS-2023-10-01"], "crs": "EPSG:21897", "target_dir": "/p/projects/eubucco/other_projects/transit_access/data/", "boundary_path": "/p/projects/eubucco/other_projects/transit_access/data/bounds/bog_bound.gpkg", "h3_res": 10, "start_time": "06:00", "end_time": "10:00"}' | python run.py
