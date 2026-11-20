from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def graficar_mapas_error(
    X1_grid: np.ndarray,
    X2_grid: np.ndarray,
    resultados_por_c: dict[float, dict],
    ruta_salida: str = "resultados/mapa_error_reconstruccion.png",
) -> None:
    """
    Exporta una figura de dos paneles con la misma escala de colores (vmin, vmax),
    ejes y unidades normalizadas para comparar los valores de c.
    """
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    valores_c = list(resultados_por_c.keys())
    forma_malla = X1_grid.shape

    mapas = {
        c: resultados_por_c[c]["estadisticas"]["norma_error_senal"].reshape(forma_malla)
        for c in valores_c
    }

    vmin = 0.0
    vmax = max(float(np.max(m)) for m in mapas.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True, constrained_layout=True)

    for ax, c in zip(axes, valores_c):
        mapa = mapas[c]
        kappa = resultados_por_c[c]["diagnostico"]["kappa_2"]
        im = ax.pcolormesh(
            X1_grid,
            X2_grid,
            mapa,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"c = {c} ($\\kappa_2 = {kappa:.1f}$)")
        ax.set_xlabel("Componente $x_1$ [u. norm.]")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Componente $x_2$ [u. norm.]")
    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.03)
    cbar.set_label("Norma euclidiana del error $\\|\\hat{\\mathbf{x}} - \\mathbf{x}\\|_2$ [u. norm.]")

    fig.suptitle("Comparación del error absoluto de reconstrucción bajo el mismo ruido ($\\sigma = 0.01$)", fontsize=12)
    fig.savefig(ruta_salida, dpi=300)
    plt.close(fig)
