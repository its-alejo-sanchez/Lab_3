import numpy as np


def analizar_matriz(A: np.ndarray) -> dict:
    """
    Calcula autovalores/autovectores (eigh), QR y SVD, verificando identidades
    y obteniendo el número de condición espectral kappa_2 = sigma_max / sigma_min.
    """
    if A.shape != (2, 2) or not np.all(np.isfinite(A)):
        raise ValueError("A debe ser una matriz finita de forma (2, 2).")

    autovalores, V_eig = np.linalg.eigh(A)
    idx = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx]
    V_eig = V_eig[:, idx]
    disc_eig = float(np.max(np.abs(A @ V_eig - V_eig * autovalores)))

    Q, R = np.linalg.qr(A)
    disc_qr_identidad = float(np.max(np.abs(Q.T @ Q - np.eye(2))))
    disc_qr_reconstruccion = float(np.max(np.abs(Q @ R - A)))

    U, S, Vt = np.linalg.svd(A)
    disc_svd = float(np.max(np.abs((U * S) @ Vt - A)))

    kappa_2 = float(S[0] / S[1])

    return {
        "autovalores": autovalores,
        "autovectores": V_eig,
        "Q": Q,
        "R": R,
        "U": U,
        "valores_singulares": S,
        "Vt": Vt,
        "kappa_2": kappa_2,
        "disc_eig": disc_eig,
        "disc_qr_ort": disc_qr_identidad,
        "disc_qr_rec": disc_qr_reconstruccion,
        "disc_svd": disc_svd,
    }


def evaluar_reconstruccion(
    A: np.ndarray,
    X_real: np.ndarray,
    B: np.ndarray,
    X_hat: np.ndarray,
    umbral_error: float = 0.1,
) -> dict:
    """
    Calcula errores, residuos, estadísticas por componente (eje 1) y por señal (eje 0),
    usando broadcasting para verificar la desviación estándar con ddof=0.
    """
    error = X_hat - X_real
    residuo = (A @ X_hat) - B

    media_error_comp = np.mean(error, axis=1)
    std_error_comp = np.std(error, axis=1, ddof=0)

    error_centrado = error - media_error_comp[:, np.newaxis]
    std_verificada = np.sqrt(np.mean(error_centrado**2, axis=1))
    assert np.allclose(std_error_comp, std_verificada), "Discrepancia en cálculo por broadcasting."

    norma_error_senal = np.linalg.norm(error, axis=0)
    norma_residuo_senal = np.linalg.norm(residuo, axis=0)

    mascara_supera = norma_error_senal > umbral_error
    clasificacion = np.where(mascara_supera, "Supera umbral", "Aceptable")
    proporcion_supera = float(np.mean(mascara_supera))

    idx_max = int(np.argmax(norma_error_senal))
    error_max = float(norma_error_senal[idx_max])
    senal_peor = X_real[:, idx_max].copy()
    residuo_max = float(np.max(norma_residuo_senal))

    return {
        "error": error,
        "residuo": residuo,
        "media_error_comp": media_error_comp,
        "std_error_comp": std_error_comp,
        "norma_error_senal": norma_error_senal,
        "mascara_supera": mascara_supera,
        "clasificacion": clasificacion,
        "proporcion_supera": proporcion_supera,
        "idx_max": idx_max,
        "error_max": error_max,
        "senal_peor": senal_peor,
        "residuo_max": residuo_max,
    }
