FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

# CPU-only torch build: the default PyPI wheel bundles several GB of CUDA
# libraries this project never uses.
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY *.py ./
COPY card_data/compiled_data.csv ./card_data/compiled_data.csv

ENTRYPOINT ["python", "train.py"]

# TODO: Add CMD to specify default arguments for training, e.g., dataset path, hyperparameters, etc.
#CMD ["--help"]
