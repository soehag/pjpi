"""Demo: transformation utilities

Run as script to visualize forward/backward transforms and derivatives.
"""

import numpy as np
import matplotlib.pyplot as plt
import gnmesh.meshtools.transformation as tF


def main():
    x = np.linspace(0.1, 9.9, 400)

    transforms = [
        tF.MultiplicativeTransformation(0.5),
        tF.PowerTransformation(2.0),
        tF.LogarithmicBarrierTransformationTwoSided(0.0, 10.0),
    ]

    fig, axs = plt.subplots(len(transforms), 3, figsize=(10, 3 * len(transforms)), constrained_layout=True)
    if transforms and axs.ndim == 1:
        # Keep the indexing logic uniform when there is only one transform.
        axs = axs[np.newaxis, :]
    fig.suptitle("Transformation utilities: forward/backward and derivatives", fontsize=12)

    def format_transform_name(tr):
        """Return a readable name and parameter line for a transformation.

        The returned string contains the class name on the first line and
        the key parameter(s) on the second line (if present).
        """
        name = type(tr).__name__
        params = None
        # These names are intentionally derived from the current private fields used by the
        # transformation classes in this demo; add a branch here when a new transform type appears.
        if hasattr(tr, "_power"):
            params = f"power={getattr(tr, '_power')}"
        elif hasattr(tr, "_multiplier"):
            params = f"multiplier={getattr(tr, '_multiplier')}"
        elif hasattr(tr, "_lower_barrier") and hasattr(tr, "_upper_barrier"):
            params = f"lower={getattr(tr, '_lower_barrier')}, upper={getattr(tr, '_upper_barrier')}"
        elif hasattr(tr, "_barrier"):
            params = f"barrier={getattr(tr, '_barrier')}"

        if params:
            return f"{name}\n{params}"
        return name

    for i, tr in enumerate(transforms):
        # forward/backward
        y = tr.forward(x)
        inv = tr.backward(y)

        axs[i, 0].plot(x, y, label="transformed")
        axs[i, 0].plot(x, x, label="identity")
        axs[i, 0].plot(x, inv, '--', label="inverse")
        axs[i, 0].legend()
        # title the row with the transformation class name
        # use a bit more padding so the two-line title has breathing room
        axs[i, 0].set_title(format_transform_name(tr), pad=14, fontsize=10)

        # derivative forward: plot analytic and the transformation-provided numeric derivative
        tr._derivative = "analytic"
        df = tr.derivative_forward(x)
        axs[i, 1].plot(x, df, label="analytic")
        # request numeric derivative from the transform (some classes support this)
        tr._derivative = "numerical"
        try:
            df_num = tr.derivative_forward(x)
            axs[i, 1].plot(x, df_num, linestyle='--', label="numeric")
        except Exception:
            # fallback: transformation may not support numeric derivative
            pass
        axs[i, 1].set_title("d/dx forward")
        axs[i, 1].legend()

        # derivative backward: analytic and numeric (if available)
        tr._derivative = "analytic"
        db = tr.derivative_backward(x)
        axs[i, 2].plot(x, db, label="analytic")
        tr._derivative = "numerical"
        try:
            db_num = tr.derivative_backward(x)
            axs[i, 2].plot(x, db_num, linestyle='--', label="numeric")
        except Exception:
            pass
        axs[i, 2].set_title("d/dx backward")
        axs[i, 2].legend()

    plt.show()


if __name__ == "__main__":
    main()
