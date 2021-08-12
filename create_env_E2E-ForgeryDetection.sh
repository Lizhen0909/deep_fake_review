conda create --channel conda-forge --name E2E-ForgeryDetection python=3.5  -y
conda run -n E2E-ForgeryDetection  pip install -r E2E-ForgeryDetection/requirements.txt
conda run -n E2E-ForgeryDetection  pip install ipykernel

