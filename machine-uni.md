
# Connection machine  aux machines Alan GPU Cluster:

## 1. Connection VPN Unif

## 2 . Connection SSH

- host: ssh phockers@10.19.99.66
- password: chess1234

### Documentation
- [alan-cluster](https://github.com/montefiore-ai/alan-cluster)
- [slurm](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html)
- [tuto](https://vsoch.github.io/lessons/sherlock-jobs/)

### Commande Slurn
-Watch jobs:
```console
watch -n 1 squeue -u phockers
```
- run
```console
    sbatch run.sh
```
- reset git:
```console
  git fetch
  git reset --hard origin/master
```
- Commande pour retrouver Chess
```console
 squeue | grep Chess
```
- Voir si le jobs est dans la queue:
```console
squeue
```
- Supprimer un jobs:
```console
scancel --name=jobs_name
```
- Supprimer un job: deeepEvaluator:
```console
scancel --name=Chess-deepEvaluator
```
# file.sh

Pour lancer la generation des dataset: generateDataset.sh
pour lancer le code deepEvaluator : run.sh


# Run le Slurn

- Runner un batch nommé nom.sh: sbatch nom.sh
- on peut voir le résultat console dans un fichier txt précisé dans le bash, ici res_conda_gpu_deepEvaluator.txt


# Exemple de nom.sh
```console
  #!/bin/bash
  #
  #SBATCH --job-name=Chess-deepEvaluator
  #SBATCH --output=res-run.txt
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=1
  #SBATCH --ntasks-per-node=1
  #SBATCH --time=0-05:00:00
  #SBATCH --mem-per-cpu=32g
  #SBATCH --partition=all
  #SBATCH --gres=gpu:1

  python3 deepEvaluator.py
```



# TODO (mis à jour) :

on a l'impression que l'éxecution est plus lente que quand on lance une commande directe dans la console, peut être du à moins de CPU et pas de GPU utilisée pour l'instant
à investiguer.
