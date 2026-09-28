# RAG Chat

Chat con un LLM locale sui tuoi documenti.  
Usa LangChain, FAISS e llama-cpp-python per fornire risposte contestualizzate usando il retrieval-augmented generation (RAG).

---

## Prerequisiti

- **uv** (gestore di pacchetti e ambienti virtuali Python) — [installazione](https://docs.astral.sh/uv/getting-started/installation/)
- **Python 3.12** (vedi `.python-version`)

---

## 1. Installazione di Python e creazione del virtual environment

Il progetto richiede **Python 3.12** (specificato nel file `.python-version`).

### Installare Python 3.12

Se non è già presente, installalo con uv:

```bash
uv python install 3.12
```

### Creare il virtual environment

Nella cartella del progetto:

```bash
uv sync
```

Questo comando:

1. Installa Python 3.12 (se non presente) in base a `.python-version`
2. Crea il virtual environment in `.venv`
3. Installa tutte le dipendenze dichiarate in `pyproject.toml`

---

## 2. Avvio del programma

```bash
uv run ragchat
```

`uv run` sincronizza automaticamente il virtual environment (se necessario) e poi avvia l'interfaccia grafica Tkinter.

---

## 3. Avvio offline (senza accesso alla rete)

```bash
uv run --no-sync ragchat
```

Il flag `--no-sync` impedisce a uv di tentare di sincronizzare dipendenze o contattare repository remoti, permettendo l'avvio anche senza connessione a internet (assicurandosi che il virtual environment esista già).

---

## 4. Compilazione dell'eseguibile (Windows)

Per creare un file `.exe` autonomo utilizzando [PyInstaller](https://pyinstaller.org/), senza aggiungere PyInstaller alle dipendenze del progetto:

```bash
uv run --with pyinstaller pyinstaller ragchat.spec --noconfirm
```

### Spiegazione

| Elemento | Descrizione |
|---|---|
| `uv run --with pyinstaller` | Esegue PyInstaller in un ambiente temporaneo che include il progetto e le sue dipendenze, **senza** modificare `pyproject.toml` |
| `ragchat.spec` | File di configurazione PyInstaller presente nella radice del progetto — contiene hidden import e `collect_all` necessarie per i moduli lazy e i DLL nativi |
| `--noconfirm` | Sovrascrive l'output senza chiedere conferma |

### Output

- **Posizione**: `dist/ragchat.exe`
- **Dimensione**: ~415 MB
- **Opzioni build**: `--onefile` (singolo file), `--windowed` (nessuna console)

### Costruzione da zero (senza file `.spec`)

Se il file `ragchat.spec` non esiste, si può generarlo con:

```bash
uv run --with pyinstaller pyinstaller \
  --onefile --windowed --name "ragchat" \
  --paths src \
  --collect-all ragchat \
  --collect-all llama_cpp \
  --collect-all docling \
  --hidden-import "langchain_community.llms" \
  --hidden-import "langchain_community.llms.llamacpp" \
  --hidden-import "langchain_community.llms.huggingface_pipeline" \
  --hidden-import "langchain_text_splitters.character" \
  --hidden-import "docling.document_converter" \
  --hidden-import "docling.datamodel.base_models" \
  src/ragchat/__main__.py
```

> `src/ragchat/__main__.py` è il punto d'ingresso dell'applicazione e contiene:
> ```python
> from ragchat import main
> if __name__ == "__main__":
>     main()
> ```

### Note sulla build

- I DLL del runtime VC++ (`vcruntime140.dll`, `msvcp140.dll`, `vcruntime140_1.dll`) sono inclusi nel pacchetto tramite `--collect-all ragchat` e caricati automaticamente all'avvio tramite `ragchat.vendor`.
- Il modulo `langchain_community.llms` usa `__getattr__` per importare dinamicamente le classi LLM. I moduli sottostanti (`llamacpp`, `huggingface_pipeline`) sono aggiunti come hidden import per assicurare che siano disponibili a runtime.
- Alcuni warning su moduli opzionali (`nltk`, `tiktoken`, `spacy`, `transformers.AutoTokenizer`) sono normali: corrispondono a dipendenze non utilizzate dal percorso principale (modelli GGUF).
