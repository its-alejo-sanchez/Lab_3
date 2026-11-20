import numpy as np


def construir_matriz_sensores(c: float, tol_singular: float = 1e-12) -> np.ndarray:
    """Construye A(c) = [[1, c], [c, 1]] de forma (2, 2) validando no singularidad."""
    if not np.isfinite(c):
        raise ValueError("El parámetro 'c' debe ser finito.")
    if np.isclose(abs(c), 1.0, atol=tol_singular):
        raise ValueError(f"La matriz A(c={c}) es singular (|c| = 1) y no es invertible.")
    return np.array([[1.0, float(c)], [float(c), 1.0]], dtype=np.float64)


def generar_malla_senales(n_puntos: int = 21, limite: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera malla de n_puntos x n_puntos en [-limite, limite].
    Convención: señales por columnas -> X tiene forma (2, n_puntos**2).
    """
    if n_puntos < 2:
        raise ValueError("n_puntos debe ser >= 2.")
    eje = np.linspace(-limite, limite, n_puntos)
    X1_grid, X2_grid = np.meshgrid(eje, eje, indexing="xy")
    X = np.vstack([X1_grid.ravel(), X2_grid.ravel()])
    return X, X1_grid, X2_grid


def generar_ruido(forma: tuple[int, int], semilla: int = 7, desv_std: float = 0.01) -> np.ndarray:
    """Genera una realización única de ruido gaussiano N(0, desv_std^2)."""
    rng = np.random.default_rng(semilla)
    return rng.normal(loc=0.0, scale=desv_std, size=forma)


def _validar_matriz_y_senales(A: np.ndarray, M: np.ndarray, nombre_M: str = "X") -> None:
    if A.shape != (2, 2):
        raise ValueError(f"Se esperaba A de forma (2, 2), se recibió {A.shape}.")
    if M.ndim != 2 or M.shape[0] != 2:
        raise ValueError(f"Se esperaba {nombre_M} de forma (2, N), se recibió {M.shape}.")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(M))):
        raise ValueError(f"A y {nombre_M} deben contener valores finitos.")


def simular_lecturas(A: np.ndarray, X: np.ndarray, ruido: np.ndarray) -> np.ndarray:
    """Calcula B = A @ X + ruido sin modificar X."""
    _validar_matriz_y_senales(A, X, "X")
    if ruido.shape != X.shape or not np.all(np.isfinite(ruido)):
        raise ValueError("El ruido debe ser finito y tener exactamente la misma forma que X.")
    return (A @ X) + ruido


def reconstruir_senales(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Resuelve A @ X_hat = B usando numpy.linalg.solve sin calcular inversa."""
    _validar_matriz_y_senales(A, B, "B")
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError as exc:
        raise ValueError("El sistema lineal es singular.") from exc
