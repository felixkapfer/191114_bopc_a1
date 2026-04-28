import numpy as np
from julia_par import (
    compute_julia_set_sequential,
    compute_julia_in_parallel,
    BENCHMARK_C,
)


def main():
    # Use a small problem so the test is fast
    size = 80
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5
    c = BENCHMARK_C

    ref = compute_julia_set_sequential(xmin, xmax, ymin, ymax, size, size, c)

    cases = [
        # (patch, nprocs)
        (1, 1), (1, 2),
        (10, 1), (10, 2),
        (24, 1), (24, 2),     # patch does not divide size -> edge patches
        (size, 1), (size, 2), # one giant patch covering everything
        (7, 2),               # awkward size
    ]
    failures = 0
    for patch, nprocs in cases:
        par = compute_julia_in_parallel(
            size, xmin, xmax, ymin, ymax, patch, nprocs, c)
        ok = np.allclose(ref, par)
        print(f"patch={patch:3d}, nprocs={nprocs}: "
              f"{'OK' if ok else 'MISMATCH'}, "
              f"max diff = {np.max(np.abs(ref - par)):.2e}")
        if not ok:
            failures += 1

    print()
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} test(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())