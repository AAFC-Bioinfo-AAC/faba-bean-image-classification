[English version of this document](./README.md)  

[instructions pour l'installation et l'utilisation](/instructions.md)  


# Pipeline d'extraction de caractéristiques de féverole à partir d'images WGRF-féverole

## Aperçu

Ce code est une pipeline d'extraction des caractéristiques de féverole afin d'extraire la dimension, la forme et la couleur des graines de féverole dans le fichier .csv à partir des images de féveroles. Il présente une méthodologie pour la segmentation d'images et l'extraction de caractéristiques à l'aide de techniques avancées d'apprentissage profond et de traitement d'images. Le [Segment Anything Model 2.1](https://github.com/facebookresearch/sam2/blob/main/README.md) (SAM2.1) a été utilisé pour une segmentation précise, tandis que [OpenCV](https://docs.opencv.org/4.x/d7/dbd/group__imgproc.html), [Scikit-Image](https://scikit-image.org/) et [Matplotlib-colors](https://matplotlib.org/stable/gallery/color/named_colors.html) sont utilisés pour analyser les propriétés dimensionnelles, spatiales, de forme et de couleur des graines segmentées. Le pipeline donne également le nombre de graines dans une image et des images binaires annotées. Le pipeline a été spécifiquement développé en fonction des coordonnées spatiales des graines de féverole, de la carte de couleur, de l'étiquette, de la règle et de la pièce.

### Images de féverole

Les images des féveroles ont été prises conformément au protocole d'exploitation normalisé (figure 1).

<img src="https://gccode.ssc-spc.gc.ca/lethbridge-carsu/wgrf-cloud-phenomics/faba-bean-image-classification/-/raw/main/harpreet_scripts/Images/Faba-Seed-CC_Vf1-1-2.JPG » alt="Figure 1 » width="200">

Graphique 1. Exemple d'images de féverole Vf1-1-2 (forme de l'image = 6000, 4000, 3) avec graines de féverole, carte de couleur, pièce de monnaie, étiquette et règle 

### Modèle Segmentanything 2.1 (MetaAI) utilisé pour la segmentation d'images

[Segment Anything Model 2](https://ai.meta.com/sam2/) (SAM 2.1) est un modèle de segmentation avancé conçu pour fonctionner de manière transparente avec les images et les vidéos, traitant une seule image comme une vidéo d'une seule image. Ce travail introduit une nouvelle tâche, un nouveau modèle et un nouvel ensemble de données visant à améliorer la performance de la segmentation. SAM 2 entraîné sur un ensemble de données SA-V offre de solides performances dans un large éventail de tâches. Dans la segmentation d'image, le modèle SAM2 est plus précis et 6 fois plus rapide que le modèle Segment Anything (SAM). 

## 🔥 Un aperçu rapide

<img src="https://gccode.ssc-spc.gc.ca/lethbridge-carsu/wgrf-cloud-phenomics/faba-bean-image-classification/-/raw/main/harpreet_scripts/Images/SAM2.1_Flowchart.png » alt="Figure 2 » width="800">

Figure 2 : Organigramme du pipeline d'extraction des caractéristiques de la féverole

## 📝 Détails des étapes **(Figure 2)** :

1. **Étape 1 :** Les images/images sont utilisées comme entrée et le modèle SAM2.1 génère les masques binaires (.png) et le fichier de métadonnées (.csv) pour chaque image dans le répertoire de sortie SAM

2. **Étape 2 :** Le répertoire de sortie SAM (de l'étape 2) est utilisé comme entrée pour cette étape et l'analyse des données, l'extraction des caractéristiques à l'aide de la bibliothèque d'images sci-kit et l'ingénierie des caractéristiques donne le fichier .csv avec les caractéristiques dimensionnelles et de forme dans un autre répertoire de sortie FE

3. **Étape 3 :** Le répertoire de sortie FE (à partir de l'étape 2) et les images (utilisées comme entrée à l'étape 1) seront utilisés comme entrées pour cette étape et les étiquettes de couleur et les valeurs RVB seront extraites à l'aide de la bibliothèque colormath pour donner .csv fichier dans le même répertoire de sortie finale FE (à partir de l'étape 2).

## 📚 Fichiers de résultats 

Après avoir exécuté le pipeline d'extraction de caractéristiques de féveroles, il y aura 2 répertoires de résultats-
1. **Output dir SAM** contiendra des sous-dossiers (Faba-Seed-CC_Vf_N-N_N) avec des masques (N.png) et un fichier de métadonnées (metadata.csv) pour chaque image. 
2. **Output dir FE** contiendra :
a. Le fichier .csv des caractéristiques dimensionnelles et de forme (Fava_bean_Features_extraction.csv)
b. Le fichier .csv des valeurs dimensionnelles, des formes, des valeurs RVB et des noms de couleur (FE_Color.csv)
c. Nombre de semences (.xlsx) (Count.xlsx semences)
d. Image binaire annotée (.png) avec contours autour des haricots (Faba-Seed-CC_Vf_N-N_N_combined_mask.png) 

Les caractéristiques qui ont été extraites de ce pipeline sont les suivantes :
1. **Caractéristiques dimensionnelles (19)** : Area_mm2_SAM, Length_mm_SAM, Width_mm_SAM, perimeter_mm_SAM, centroïde-0, centroïde-1, bbox-0, bbox-1, bboîx-2, bboîx-3, Area_pix_SAM, excentricité, equivalent_diameter_area, périmètre, solidité, area_convex, étendue, longueur de l'axe majeur (pix)_SAM, longueur de l'axe mineur (pix)_SAM, Aspect_Ratio, rondeur, compacité, Circularity_SAM
2. **Caractéristiques de forme (4)** : Forme, Facteur de forme1, Facteur de forme2, Facteur de forme3, Facteur de forme4
3. **Couleur (2)** : valeur RVB, color_seeds
4. **Nombre de graines** : Nombre de graines sur l'image



