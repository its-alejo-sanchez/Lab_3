import numpy as np
from src.modelo import (
    construir_matriz_sensores,
    generar_malla_senales,
    generar_ruido,
    simular_lecturas,
    reconstruir_senales,
)
from src.diagnostico import analizar_matriz, evaluar_reconstruccion
from src.visualizacion import graficar_mapas_error


def ejecutar_experimento():
    valores_c = [0.2, 0.99]
    X_real, X1_grid, X2_grid = generar_malla_senales(n_puntos=21, limite=1.0)
    X_respaldo = X_real.copy()  

    ruido = generar_ruido(forma=X_real.shape, semilla=7, desv_std=0.01)

    resultados = {}
    for c in valores_c:
        A = construir_matriz_sensores(c)
        B = simular_lecturas(A, X_real, ruido)
        X_hat = reconstruir_senales(A, B)

        diag = analizar_matriz(A)
        est = evaluar_reconstruccion(A, X_real, B, X_hat, umbral_error=0.1)
        resultados[c] = {"diagnostico": diag, "estadisticas": est}

    assert np.array_equal(X_real, X_respaldo), "El arreglo original X_real fue alterado."

    print("\n=== TABLA 1: DIAGNÓSTICO MATRICIAL ===")
    print(f"{'c':<6} | {'Autovalores (λ)':<18} | {'Val. Singulares (σ)':<20} | {'κ₂':<8} | {'Max Disc. Id.':<14}")
    print("-" * 78)
    for c in valores_c:
        d = resultados[c]["diagnostico"]
        max_disc = max(d["disc_eig"], d["disc_qr_ort"], d["disc_qr_rec"], d["disc_svd"])
        lams = f"[{d['autovalores'][0]:.2f}, {d['autovalores'][1]:.2f}]"
        sigs = f"[{d['valores_singulares'][0]:.2f}, {d['valores_singulares'][1]:.2f}]"
        print(f"{c:<6.2f} | {lams:<18} | {sigs:<20} | {d['kappa_2']:<8.2f} | {max_disc:<14.2e}")

    print("\n=== TABLA 2: ESTADÍSTICAS DE RECONSTRUCCIÓN ===")
    print(f"{'c':<6} | {'Media err (x1, x2)':<22} | {'Std err (x1, x2)':<22} | {'Err Máx':<9} | {'% > 0.1':<8} | {'Res. Máx':<10}")
    print("-" * 92)
    for c in valores_c:
        e = resultados[c]["estadisticas"]
        med = f"({e['media_error_comp'][0]:+.4f}, {e['media_error_comp'][1]:+.4f})"
        std = f"({e['std_error_comp'][0]:.4f}, {e['std_error_comp'][1]:.4f})"
        print(
            f"{c:<6.2f} | {med:<22} | {std:<22} | {e['error_max']:<9.4f} | "
            f"{e['proporcion_supera']*100:<7.1f}% | {e['residuo_max']:<10.2e}"
        )

    graficar_mapas_error(X1_grid, X2_grid, resultados)
    print("\nFigura guardada en: resultados/mapa_error_reconstruccion.png")


if __name__ == "__main__":
    ejecutar_experimento()
