# Datasets de OjoScan / Eye-Retina AI

Fuentes de todos los datos usados o revisados en el proyecto. Actualizado: 2026-11-14.
Copias locales en `~/TemrariosSpace/datasets/`.

## 1. Usados en modelos que están en producción

| Para qué | Dataset | Enlace | Licencia | Qué se usó |
|---|---|---|---|---|
| **Vasos de retina** (U-Net) | FIVES — 800 fondos de ojo con vasos marcados | https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169 · descarga directa: https://ndownloader.figshare.com/files/34969398 | CC BY 4.0 | Entrenamiento y prueba (510 / 90 / 200) |
| Vasos de retina (prueba externa) | HRF — High-Resolution Fundus, 45 fondos | https://www5.cs.fau.de/research/data/fundus-images/ | CC BY 4.0 | Solo prueba externa |
| **Ojo externo**: enrojecimiento / conjuntivitis / blefaritis / catarata / normal | Pink-Eye-z | https://huggingface.co/datasets/chaudharyusman/Pink-Eye-z | Ver página | Entrenamiento del clasificador de ojo |
| **Ojo externo**: pterigión | pterygium2class | https://huggingface.co/datasets/mostafasmart/pterygium2class | Ver página | Entrenamiento del clasificador de ojo |
| **Anemia por conjuntiva** (párpado) | Eyes-defy-anemia — 218 ojos con hemoglobina (India e Italia) | Original: https://ieee-dataport.org/documents/eyes-defy-anemia (DOI 10.21227/t5s2-4j73, pide cuenta) · copia usada: https://huggingface.co/datasets/Yahaira/anemia-eyes | Investigación | Clasificador de anemia del párpado |
| Anemia: encontrar la conjuntiva | Anemia-seg (máscaras de conjuntiva) | https://huggingface.co/datasets/bharath56858/Anemia-seg | Apache 2.0 | Recortar la conjuntiva en la foto |
| **Anemia por la palma** | Ghana — palma, 710 niños de 6 a 59 meses con hemoglobina | https://data.mendeley.com/datasets/ccr8cm22vz/1 | CC BY 4.0 | Clasificador de la palma |
| **Parálisis facial** | PalsyNet — 49 personas (27 con parálisis) | https://huggingface.co/datasets/lili3329/palsynet-5230 | CC BY 4.0 | Asimetría de la cara |
| **Ictericia** | Jaundice_Dataset | https://huggingface.co/datasets/Bleachcarte/Jaundice_Dataset | CC BY-NC 4.0 (no comercial) | Solo las 40 fotos de cara completa con ictericia |
| Caras sanas de control (parálisis e ictericia) | FFHQ — Flickr-Faces-HQ (NVIDIA) | https://github.com/NVlabs/ffhq-dataset | CC BY-NC-SA 4.0 (no comercial) | 907 caras para parálisis y 1,498 para ictericia |

## 2. Descargados para experimentos (no están en la app)

| Dataset | Enlace | Licencia | Resultado |
|---|---|---|---|
| Ghana — conjuntiva | https://data.mendeley.com/datasets/nt7r8hv2pz/1 | CC BY 4.0 | Comparado con Eyes-defy-anemia |
| Ghana — uñas | https://data.mendeley.com/datasets/2xx4j3kjg2/1 | CC BY 4.0 | AUC 0.62; muy bajo, no se usó |
| OCT2017 (Kermany) — ~84 mil OCT | https://data.mendeley.com/datasets/rscbjbr9sj/2 | CC BY 4.0 | Descargado; aún sin modelo |
| Acné / rosácea / piel normal | https://huggingface.co/datasets/Neperl/skin-disease-acne-rosacea-normal | Ver página | Rosácea descartada (AUC 0.52) |

## 3. Encía y perfil médico (agregados el 2026-11-14)

| Para qué | Dataset | Enlace | Licencia | Qué se usó |
|---|---|---|---|---|
| **Reconocer los dientes** (para encontrar la encía) | Teeth Segmentation Intraoral (Dentalai) — 2,495 fotos con dientes marcados | https://huggingface.co/datasets/Frallex/Teeth_Segmentation_Intraoral | MIT | U-Net de dientes, Dice 0.92 en prueba |
| **Color de encía sana** (referencia) | Dental IntraOral Gingivitis — 1,096 fotos, grado de gingivitis por diente | Original: https://data.mendeley.com/datasets/3253gj88rr/1 · copia usada: https://huggingface.co/datasets/ekacare/IntraOral_Gingivitis_Image_Captioning | CC BY 4.0 | 261 franjas de encía sana (grado 0) del maxilar; prueba del medidor: AUC 0.73 sana vs inflamada |
| **Grasa corporal** | bodyfat.csv — 252 hombres (de `~/Deep_Learning`) | https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset (original: http://lib.stat.cmu.edu/datasets/bodyfat) | Uso libre (StatLib) | Regresión con cinta métrica, error ±3.6 % |
| **Riesgo coronario** | UCI Heart Disease, Cleveland — 303 pacientes (el `heart.xls` de `analysis.ipynb`) | https://archive.ics.uci.edu/dataset/45/heart+disease | CC BY 4.0 | Logística con cuestionario + presión, AUC 0.85 |
| (no se usó) | Breast Cancer Wisconsin (de `~/Deep_Learning`) | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic | CC BY 4.0 | Requiere medidas de biopsia: no aplica a una app |

`Class_ID` del dataset de gingivitis = grado + 2 (verificado con los textos de los periodoncistas).
Ninguno de estos datos trae hemoglobina: **la anemia por encía no está validada** (es experimental).

## 4. Revisados y descartados

| Dataset / estudio | Enlace | Por qué no |
|---|---|---|
| Anemia por la encía — 300 fotos (Cureus 2024) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11260651/ | No publican los datos; habría que pedirlos a la autora |
| Anemia por los labios — 138 personas (Frontiers in Big Data 2023) | https://pmc.ncbi.nlm.nih.gov/articles/PMC10655138/ | Los datos solo se dan por correo a los autores |
| Video de la yema del dedo — 206 personas (Zenodo) | https://zenodo.org/records/24135340 | Acceso restringido y solo personas sin anemia |
| GingivitisSeg — máscaras de inflamación de encía | https://arxiv.org/abs/2611.15184 · https://github.com/zeynepozdemir/GingivitisSeg | El repositorio está vacío y el DOI de Zenodo no existe |
| AlphaDent — 1,320 fotos de boca | https://github.com/ZFTurbo/AlphaDent · https://zenodo.org/records/16582489 | Solo marca problemas de los dientes, no la encía; además pesa 4.9 GB |
| MouthCare intraoral-sample | https://huggingface.co/datasets/MouthCare/intraoral-sample | Acceso restringido |
| CODE — mucosa oral | https://figshare.com/articles/dataset/_b_CODE_-_b_i_Comprehensive_Oral_mucosa_Database_with_Explanations_i_/30550889 | Etiquetado para cáncer oral, no anemia |
| MeMoSA — 30 mil fotos de mucosa oral | https://www.nature.com/articles/s41597-026-06998-7 | Lesiones orales, no anemia |
| TCM-Tongue — 6,719 lenguas | https://arxiv.org/pdf/2507.18288 | Diagnósticos de medicina china, no anemia |
| Ghana — fuente original con país y hemoglobina | IEEE DataPort | Pide cuenta |
| Lista de retina en `datasets/README.md` (IDRiD, DRIVE, CHASE, APTOS, etc.) | ver `~/TemrariosSpace/datasets/README.md` | Para fases siguientes de retina |

> Ojo con las licencias: Jaundice_Dataset y FFHQ son **no comerciales**. Si la app se vende,
> esos dos modelos (ictericia y el control de caras sanas) hay que reentrenarlos con otros datos.
