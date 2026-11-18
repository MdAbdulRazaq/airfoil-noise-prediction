$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
& .\.venv\Scripts\python.exe -m pytest -q
