"""Encía: la máscara toma lo rosado de la boca y deja fuera dientes, sombra y labio."""
import numpy as np

from src.deployment.inference_encia import blanco_dientes, mascara_encia


def boca_sintetica():
    img = np.zeros((100, 200, 3), np.float32)
    img[:, :] = (0.75, 0.45, 0.42)          # labio/piel alrededor
    img[30:70, 20:180] = (0.85, 0.45, 0.50)  # encía rosada
    img[45:70, 40:160] = (0.93, 0.92, 0.88)  # dientes
    img[65:70, 20:180] = (0.05, 0.03, 0.03)  # sombra
    region = np.zeros((100, 200), bool)
    region[30:70, 20:180] = True             # abertura de la boca
    dientes = np.zeros((100, 200), bool)
    dientes[45:70, 40:160] = True
    return img, region, dientes


def test_encia_sin_dientes_ni_sombra():
    img, region, dientes = boca_sintetica()
    m = mascara_encia(img, region, dientes)
    assert m[35, 100] and m[50, 25]          # encía arriba y al lado
    assert not m[55, 100]                    # diente
    assert not m[67, 100]                    # sombra
    assert not m[10, 100]                    # fuera de la boca (labio)


def test_blanco_de_los_dientes():
    img, _, dientes = boca_sintetica()
    b, n = blanco_dientes(img, dientes)
    assert n > 200 and np.allclose(b, (0.93, 0.92, 0.88), atol=0.02)


def test_solo_encia_de_arriba():
    from src.deployment.inference_encia import solo_sobre_dientes
    region = np.ones((10, 10), bool)
    dientes = np.zeros((10, 10), bool)
    dientes[4:6, 2:8] = True                 # dientes de arriba en filas 4-5
    m = solo_sobre_dientes(region, dientes)
    assert m[2, 5] and not m[7, 5]           # encima sí, debajo (lengua) no
    assert not m[2, 0]                       # columna sin diente: no
