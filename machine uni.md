
# Connection machine  aux machines Alan GPU Cluster:

## 1. Connection VPN Unif

## 2 . Connection SSH

- host: ssh phockers@10.19.99.66
- password: chess1234

## Documentation
- [alan-cluster](https://github.com/montefiore-ai/alan-cluster)
- [slurm](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html)
- [tuto](https://vsoch.github.io/lessons/sherlock-jobs/)

# Commande Slurn

- Voir si le jobs est dans la queue: squeue
- Supprimer un jobs: scancel --name=jobs_name

# Run le Slurn

- Runner un batch nommé nom.sh: sbatch nom.sh
- on peut voir le résultat console dans un fichier txt précisé dans le bash, ici res_conda_gpu_deepEvaluator.txt


# Exemple de nom.sh

  #!/bin/bash
  #
  #SBATCH --cpus-per-task=1        # Number of CPU cores to allocate
  #SBATCH --job-name "Chess"
  #SBATCH --mem-per-cpu=4000       # Memory to allocate in MB per allocated CPU core
  #SBATCH --gres=gpu:0             # Number of GPU's
  #SBATCH --time="7-00:00:00"      # Max execution time

  python3 deepEvaluator.py


# TODO :
on a lancé un script CHESStest qui doit faire run le NN, on ne sait pas s'il tourne vraiment car le print du début n'a tjs pas été affiché dans res_conda_gpu_deepEvaluator.txt
il faudra donc regarder plus tard si des trucs se sont bien passés.
Normalement on a accès à cuda, on a écrit un script test pour print torch.cuda.is_available et ca marchait, mais on a pas encore testé avec le network
on a l'impression que l'éxecution est plus lente que quand on lance une commande directe dans la console, peut être du à moins de CPU et pas de GPU utilisée pour l'instant
à investiguer
