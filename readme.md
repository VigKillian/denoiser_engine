# Projet Image : Débruiteur d'images GAN 

## Slides semaine 1 : https://docs.google.com/presentation/d/1ieagxBHYyKp3GInJJxnne_svQI6RG-Uf5wT322rOXIE/edit?usp=sharing

## Créer un environnement

```bash
conda create -n convdae_py37 python=3.7 -y
conda activate convdae_py37
```
## Installer les dépendances

```bash
pip install tensorflow==1.13.1 scikit-image numpy scipy matplotlib
pip install protobuf==3.20.3
```

## Lancer l’entraînement
```bash
python bsc-ConvDAE.py
```