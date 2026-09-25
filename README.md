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

Le notebook affiche au debut de l'execution le device utilise (`cuda` si un GPU est disponible, sinon `cpu`). Sur ce projet, le device detecte etait : `cuda` / `cpu` (a completer apres execution).

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

_A completer apres execution : tableau et courbes loss / accuracy / precision / recall pour les deux experiences._

| Modele | Val accuracy | Val precision | Val recall | Test accuracy |
|---|---|---|---|---|
| CNN from scratch | | | | |
| Transfert learning (ResNet18) | | | | |

**Analyse (2 a 3 paragraphes) :**

_A completer : quel modele converge le plus vite et pourquoi, quel modele a les meilleures metriques finales, ce que montrent les matrices de confusion._

## Limites et pistes d'amelioration

_A completer brievement : taille du jeu de donnees, surapprentissage observe, temps d'entrainement, pistes (fine-tuning des dernieres couches du ResNet, autres architectures, plus d'augmentation de donnees...)._

## Reproductibilite

Une graine aleatoire (`SEED = 42`) est fixee au debut du notebook pour `random`, `numpy` et `torch`, afin de garantir des resultats reproductibles.
