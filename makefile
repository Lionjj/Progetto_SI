# Nome abiente virtulae
VENV = myenv

# Installa le dipendenze
install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

# Dati preparati
prepare_data: 
	python scripts/preprocess.py

train:
	python src/train.py --epochs 10 --batch_size 32

# Esegui i test
test:
	pytest tests/

# Pulisci file temporanei e cache
clean:
	rm -rf __pycache__*.pyc $(VENV)

# Comandi disponibili
help:
	@echo "Comandi disponibili"
	@echo "	make install		- Installa le dipendenze"
	@echo "	make prepare_data	- Prepara i dataset"
	@echo "	make train			- Adestra il modello"
	@echo "	make test			- Esegui i test"
	@echo "	make clean			- Pulisci i file temporanei"