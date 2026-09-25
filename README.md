# CNN from scratch vs Transfert Learning - Cats vs Dogs

## Objectif du projet

Comparer un modele CNN entraine from scratch et un modele en transfert d'apprentissage (ResNet18 pre-entraine sur ImageNet) sur le jeu de donnees Cats vs Dogs, en suivant la loss, l'accuracy, la precision et le recall a chaque epoque, et en evaluant l'impact du transfert learning sur la convergence et la performance.

## Environnement

Installer les dependances avec :

```bash
pip install -r requirements.txt
```

## Organisation des donnees

Le jeu de donnees utilise est [Cats vs Dogs (Kaggle)](https://www.kaggle.com/c/dogs-vs-cats).

1. Telecharger le dataset depuis Kaggle.
2. Organiser les images dans un dossier `data/` a la racine du projet, avec un sous-dossier par classe (format attendu par `ImageFolder` de torchvision) :

```
data/
  train/
    cats/
      cat.0.jpg
      cat.1.jpg
      ...
    dogs/
      dog.0.jpg
      dog.1.jpg
      ...
  test/
    cats/
      ...
    dogs/
      ...
```

Le dossier `data/` n'est pas pousse sur GitHub (voir `.gitignore`).

## Verification du GPU

Le notebook affiche au debut de l'execution le device utilise (`cuda` si un GPU est disponible, sinon `mps` sur Mac Apple Silicon, sinon `cpu`). Sur cette execution, le device detecte etait `cuda` (GPU T4 sur Google Colab).

## Commandes pour entrainer

Tout se fait depuis le notebook `notebook.ipynb`, section par section :

- **Experience A (CNN from scratch)** : section 4. Parametres cles modifiables dans les cellules : `NUM_EPOCHS_A`, `optimizer_choice` (`adam` ou `sgd`), taux de dropout dans `SimpleCNN(dropout=0.5)`, le scheduler `StepLR`.
- **Experience B (Transfert learning)** : section 5. Backbone ResNet18 gele (`freeze_backbone=True`), seule la tete de classification (`model.fc`) est entrainee. Parametres cles : `NUM_EPOCHS_B`, taux de dropout dans `build_transfer_model(dropout=0.5)`, scheduler `CosineAnnealingLR`.

## Commandes pour evaluer / recharger le modele

La section 7 du notebook recharge automatiquement les checkpoints locaux :

```
checkpoints/scratch_final_best.pth
checkpoints/transfer_final_best.pth
```

et evalue les deux modeles sur le jeu de test.

## Resultats

| Modele | Val accuracy | Val precision | Val recall | Test accuracy | Test precision | Test recall |
|---|---|---|---|---|---|---|
| CNN from scratch | 0.778 | 0.915 | 0.615 | 0.774 | 0.913 | 0.606 |
| Transfert learning (ResNet18) | 0.919 | 0.934 | 0.902 | 0.913 | 0.929 | 0.894 |

**Analyse :**

Le modele en transfert learning converge beaucoup plus vite que le CNN from scratch : il atteint 91.5% d'accuracy en validation deja a la premiere epoque, alors que le CNN from scratch plafonne a 77.8% apres 6 epoques completes. Cela s'explique par le pre-entrainement sur ImageNet : ResNet18 sait deja reconnaitre des formes et textures generiques, il n'a qu'a apprendre a les combiner pour separer chats et chiens.

Sur le jeu de test, le transfert learning devance nettement le CNN from scratch sur toutes les metriques (91.3% contre 77.4% d'accuracy). La precision est proche entre les deux modeles, mais l'ecart se voit surtout sur le recall (89.4% contre 60.6%) : le CNN from scratch rate une bonne partie des vrais positifs d'une des deux classes, alors que le modele pre-entraine est a la fois precis et complet.

## Limites et pistes d'amelioration

Le CNN from scratch n'a ete entraine que sur 6 epoques, ce qui limite sa performance face a un modele pre-entraine. Son recall plus faible suggere un desequilibre dans ses predictions qui pourrait etre corrige avec plus d'epoques ou un ajustement du seuil de decision. Pour le transfert learning, une piste d'amelioration serait de degeler et d'affiner (fine-tuner) les dernieres couches du ResNet18 plutot que de n'entrainer que la tete de classification.

## Reproductibilite

Une graine aleatoire (`SEED = 42`) est fixee au debut du notebook pour `random`, `numpy` et `torch`, afin de garantir des resultats reproductibles.
