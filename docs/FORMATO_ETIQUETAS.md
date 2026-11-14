# Formatos de etiquetas para `data/raw/eyes/`

El cargador (`src/data/eye_labels.py`) detecta el formato automáticamente.
Si encuentra dos formatos a la vez, o clases que no reconoce como "eye"/"ojo",
se detiene y pide confirmarlo con `--override data.label_format=<formato>`.

## Landmarks (6 por ojo, en este orden)

| # | Nombre | Descripción |
| --- | --- | --- |
| 0 | `inner_corner` | Esquina interna (lagrimal, hacia la nariz) |
| 1 | `outer_corner` | Esquina externa |
| 2 | `upper_lid` | Punto medio del párpado superior |
| 3 | `lower_lid` | Punto medio del párpado inferior |
| 4 | `pupil` | Centro de la pupila |
| 5 | `iris` | Borde lateral del iris |

Los landmarks son opcionales: si faltan (todos o algunos), esos puntos se
excluyen de la pérdida y el modelo aprende solo la caja.

## CSV (una fila por ojo)

```csv
image,x,y,w,h,inner_corner_x,inner_corner_y,outer_corner_x,outer_corner_y,...,iris_x,iris_y
images/0001.jpg,120,88,40,18,160,97,120,97,...,135,97
images/0002.jpg,,,,,                       # fila sin caja = imagen sin ojos (negativa)
```

* Caja: `x,y,w,h` o `xmin,ymin,xmax,ymax`, en píxeles o normalizada (0-1, se detecta sola).
* Landmarks: `<nombre>_x/_y` o `lm0_x … lm5_y`.
* Columna `class`/`label` opcional: si hay varias clases, solo se usan las que contienen "eye"/"ojo".

## COCO (JSON)

`bbox = [x, y, w, h]` en píxeles y `keypoints = [x0, y0, v0, …, x5, y5, v5]`. Si
`v = 0`, el landmark se considera ausente. Si hay varias categorías, se usan las
de nombre "eye"/"ojo".

## YOLO / YOLO-pose (TXT)

`images/xxx.jpg` ↔ `labels/xxx.txt`, una línea por ojo, normalizado:

```
cls cx cy w h [x0 y0 v0 … x5 y5 v5]
```

Si hay más de una clase, se necesita un `classes.txt` con una línea "eye".

## Pascal VOC (XML)

Un `.xml` por imagen con `<object><name>eye</name><bndbox>…`. Este formato no
trae landmarks.
