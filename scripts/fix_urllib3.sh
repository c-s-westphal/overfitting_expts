#!/bin/bash
#$ -l tmem=2G
#$ -l h_rt=0:10:00
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N fix_urllib3

source /share/apps/source_files/python/python-3.9.5.source
source /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate

echo "Downgrading urllib3 to v1.x for OpenSSL compatibility..."
pip install --no-user "urllib3<2.0" --upgrade

echo "urllib3 downgraded successfully!"
