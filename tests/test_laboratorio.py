import numpy as np
import pytest
from src.modelo import (
    construir_matriz_sensores,
    generar_malla_senales,
    generar_ruido,
    simular_lecturas,
    reconstruir_senales,
)
from src.diagnostico import analizar_matriz, evaluar_reconstruccion


def test_caso_manual_sin_ruido():
    A = construir_matriz_sensores(0.2)
    X_manual = np.array([[1.0, 0.0], [-0.5, 1.0]])
    B_esperado = np.array([[0.9, 0.2], [-0.3, 1.0]])
    ruido_cero = np.zeros_like(X_manual)

    B_calc = simular_lecturas(A, X_manual, ruido_cero)
    assert np.allclose(B_calc, B_esperado, atol=1e-14)

    X_rec = reconstruir_senales(A, B_esperado)
    assert np.allclose(X_rec, X_manual, atol=1e-14)


@pytest.mark.parametrize("c_singular", [1.0, -1.0, 1.0 + 1e-15])
def test_contratos_singularidad_y_sensibilidad(c_singular):
    with pytest.raises(ValueError, match="singular"):
        construir_matriz_sensores(c_singular)


def test_c_099_no_es_singular():
    A = construir_matriz_sensores(0.99)
    assert A.shape == (2, 2)


@pytest.mark.parametrize("valor_invalido", [np.nan, np.inf, -np.inf])
def test_contratos_valores_no_finitos(valor_invalido):
    with pytest.raises(ValueError):
        construir_matriz_sensores(valor_invalido)

    A = construir_matriz_sensores(0.2)
    X_bad = np.array([[1.0, valor_invalido], [0.0, 1.0]])
    with pytest.raises(ValueError):
        simular_lecturas(A, X_bad, np.zeros((2, 2)))


def test_contratos_formas_incompatibles():
    A = construir_matriz_sensores(0.2)
    X_forma_mala = np.ones((3, 10))
    with pytest.raises(ValueError):
        simular_lecturas(A, X_forma_mala, np.zeros((3, 10)))


@pytest.mark.parametrize("c", [0.2, 0.99])
def test_identidades_matriciales(c):
    A = construir_matriz_sensores(c)
    diag = analizar_matriz(A)

    assert diag["disc_eig"] < 1e-14
    assert diag["disc_qr_ort"] < 1e-14
    assert diag["disc_qr_rec"] < 1e-14
    assert diag["disc_svd"] < 1e-14
    assert diag["kappa_2"] > 1.0


def test_estadisticas_y_mascara_manual():
    A = construir_matriz_sensores(0.2)
    X_real = np.zeros((2, 3))

    X_hat = np.array([
        [0.03, 0.06, 0.00],
        [0.04, 0.08, 0.15],
    ])
    B = A @ X_real

    est = evaluar_reconstruccion(A, X_real, B, X_hat, umbral_error=0.1)

    assert np.allclose(est["media_error_comp"], [0.03, 0.09])
    assert np.allclose(est["norma_error_senal"], [0.05, 0.10, 0.15])
    assert est["idx_max"] == 2
    assert np.isclose(est["error_max"], 0.15)
    assert np.isclose(est["proporcion_supera"], 1.0 / 3.0)
    assert list(est["clasificacion"]) == ["Aceptable", "Aceptable", "Supera umbral"]


def test_reproducibilidad_e_inmutabilidad():
    r1 = generar_ruido((2, 441), semilla=7, desv_std=0.01)
    r2 = generar_ruido((2, 441), semilla=7, desv_std=0.01)
    assert np.array_equal(r1, r2)

    X, _, _ = generar_malla_senales(21, 1.0)
    X_copia_seguridad = X.copy()
    A = construir_matriz_sensores(0.99)
    B = simular_lecturas(A, X, r1)
    X_hat = reconstruir_senales(A, B)
    est = evaluar_reconstruccion(A, X, B, X_hat)

    est["senal_peor"][0] = 999.0
    assert np.array_equal(X, X_copia_seguridad)
