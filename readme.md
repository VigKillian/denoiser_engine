# Projet Image : Débruiteur d'images GAN 

## Slides & Comptes rendus :

Les comptes rendus sont disponible dans le répertoire correspondant.

### Semaine 1 :
slides : https://docs.google.com/presentation/d/1ieagxBHYyKp3GInJJxnne_svQI6RG-Uf5wT322rOXIE/edit?usp=sharing

# Pour l'exécuter
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

Dataset: [google drive dataset link](https://drive.google.com/file/d/12OFm7Jcm7sBflKzrlH4TH7ZQA3fVA0HO/view?usp=sharing)


## Dans la structure ARM (T^T)
```bash
conda activate convdae_py37
pip install --no-cache-dir "protobuf==3.19.6"
pip install --no-cache-dir "tensorflow==2.10.*"

pip install --no-cache-dir \
  "numpy==1.21.6" \
  "scipy==1.7.3" \
  "scikit-image==0.19.3" \
  "matplotlib==3.5.*"
```

# Pour utiliser la predicteur(utiliser le modele qui est deja entraine)

Faut d'abord installer cv2
```bash
pip install opencv-python

# for arm : pip install opencv-python-headless
```