# RAG Chat Application — Documentazione Implementazione

## Overview

Applicazione Python con GUI Tkinter che permette di:
1. Caricare documenti in un database vettoriale FAISS (tramite conversione Docling)
2. Chattare con un LLM locale (gemma2-2b-it GGUF) usando RAG sui documenti indicizzati

Il progetto è gestito con **uv**, Python 3.12, struttura package `src/ragchat/`.

---

## Struttura del progetto

```
src/ragchat/
├── __init__.py                  # entry point: main() → MainApp.mainloop()
├── app.py                       # MainApp(tk.Tk) — finestra principale, layout
├── core/
│   ├── __init__.py
│   ├── embeddings.py            # HuggingFaceEmbeddings singleton (multilingual-e5-large)
│   ├── vectorstore.py           # FAISSStore — create/load/save/add/remove/retriever
│   ├── ingestor.py              # Docling → chunking → FAISS
│   ├── llm.py                   # gemma2-2b-it GGUF, download HF, bundled VC++ DLL
│   └── rag.py                   # RAGChain — retrieval + prompt + LLM
├── ui/
│   ├── __init__.py
│   ├── db_panel.py              # DBPanel(tk.Frame) — pannello sinistro
│   └── chat_panel.py            # ChatPanel(tk.Frame) — pannello destro
├── utils/
│   ├── __init__.py
│   └── metadata.py              # MetadataStore — JSON persistenza doc→chunk IDs
└── vendor/
    ├── __init__.py
    └── win_runtime/             # VC++ 2022 runtime DLL (Windows, bundled)
        ├── MSVCP140.dll
        ├── VCRUNTIME140.dll
        └── VCRUNTIME140_1.dll
```

---

## Dipendenze (`pyproject.toml`)

### Cross-platform (sempre richieste)
| Pacchetto | Scopo |
|---|---|
| `langchain` | Chain RAG, text splitter |
| `langchain-community` | FAISS loader, LlamaCpp wrapper |
| `langchain-huggingface` | `HuggingFaceEmbeddings` |
| `faiss-cpu` | Vector store |
| `docling` | Conversione documenti strutturata |
| `huggingface-hub` | Download modelli GGUF da HF |
| `sentence-transformers` | Backend embedding |

### Solo Windows (`sys_platform == 'win32'`)
| Pacchetto | Sorgente | Scopo |
|---|---|---|
| `torch` | Wheel CPU-only PyTorch per win_amd64+cp312 | Backend sentence-transformers |
| `llama-cpp-python` | Wheel precompilato CPU-only da GitHub releases v0.3.35 | Runtime GGUF locale |

### Prerequisiti di sistema — NESSUNO
Le **VC++ 2022 runtime DLL** (`MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`) sono incluse nel progetto in `src/ragchat/vendor/win_runtime/` e caricate automaticamente all'avvio. Non è necessario installare il Visual C++ Redistributable.

---

## Modelli

### LLM — gemma-2-2b-it GGUF
- **Repository HuggingFace**: `bartowski/gemma-2-2b-it-GGUF`
- **File preferito**: `gemma-2-2b-it-Q4_K_M.gguf` (fallback: `gemma-2-2b-it-Q8_0.gguf`)
- **Download**: automatico al primo utilizzo via `huggingface_hub.hf_hub_download`
- **Cache**: `~/.cache/huggingface/hub/`
- **Parametri**: `n_ctx=4096`, `temperature=0.7`, `max_tokens=512`, `n_threads=4`

### Embedding — multilingual-e5-large
- **Repository HuggingFace**: `intfloat/multilingual-e5-large`
- **Lingue**: IT, EN e 100+ altre
- **Download**: automatico via `sentence-transformers`
- **Configurazione**: `normalize_embeddings=True`, device `cpu`
- **Prefissi obbligatori**: `"query: "` per le domande utente, `"passage: "` per i chunk documenti

---

## Descrizione dei componenti

---

### `utils/metadata.py` — MetadataStore

Gestisce la persistenza JSON dei metadati associati al DB FAISS. Il file `metadata.json` risiede nella stessa cartella del DB.

**Struttura JSON**:
```json
{
  "documents": {
    "file.pdf": {
      "filename": "file.pdf",
      "chunk_ids": ["uuid1", "uuid2", ...],
      "inserted_at": "2024-01-15T10:30:00"
    }
  }
}
```

**Note implementative**:
- I chunk IDs sono stringhe UUID assegnate da LangChain FAISS (non interi come pianificato originariamente)
- Il file viene salvato automaticamente dopo ogni `add_document` e `remove_document`
- JSON corrotto: warning + reinizializzazione a struttura vuota

**API**:
| Metodo | Firma | Comportamento |
|---|---|---|
| `load()` | `→ None` | Carica da disco; init vuoto se assente/corrotto |
| `save()` | `→ None` | Serializza su disco con `indent=2`, UTF-8 |
| `add_document()` | `(name, chunk_ids) → None` | Aggiunge/sostituisce; salva |
| `remove_document()` | `(name) → list[str]` | Rimuove; restituisce chunk IDs; salva |
| `list_documents()` | `() → list[dict]` | Restituisce tutte le entrate |
| `get_document()` | `(name) → dict\|None` | Singola entrata o None |

---

### `core/embeddings.py` — Embedding singleton

**API pubblica**:
- `get_embeddings()` — singleton lazy-loaded di `HuggingFaceEmbeddings`
- `format_query(text)` — aggiunge prefisso `"query: "` per le domande utente
- `format_passage(text)` — aggiunge prefisso `"passage: "` per i chunk

---

### `core/vectorstore.py` — FAISSStore

Wrapper del DB FAISS con gestione del ciclo di vita su disco.

**Note implementative**:
- Il DB fisico viene creato solo al primo `add_documents` (`_db = None` fino ad allora)
- Il campo `langchain_core.documents.Document` è usato al posto del deprecato `langchain.schema.Document`
- I chunk IDs vengono tracciati confrontando le chiavi del `docstore._dict` prima e dopo ogni inserimento
- La rimozione di documenti ricostruisce l'intero indice FAISS dai soli chunk rimanenti (FAISS CPU non supporta delete per ID)
- Se dopo una rimozione non rimangono documenti, `_db` torna a `None`

**API**:
| Metodo | Firma | Comportamento |
|---|---|---|
| `create()` | `(db_path) → FAISSStore` | Crea cartella; `_db = None` |
| `load()` | `(db_path) → FAISSStore` | Carica da disco; `_db = None` se assente |
| `save()` | `() → None` | `save_local()`; no-op se `_db is None` |
| `add_documents()` | `(docs, filename) → int` | Crea o aggiunge; aggiorna metadati; salva; ritorna chunk inseriti |
| `remove_document()` | `(filename) → bool` | Ricostruisce DB; aggiorna metadati; `True` se rimosso |
| `list_documents()` | `() → list[dict]` | Delega a MetadataStore |
| `as_retriever()` | `(k=4) → Retriever` | `RuntimeError` se `_db is None` |
| `is_empty()` | `() → bool` | `True` se `_db is None` o docstore vuoto |

---

### `core/ingestor.py` — Pipeline di ingestione

**Pipeline** (`ingest_document(file_path, faiss_store) → int`):
1. Verifica esistenza file (`FileNotFoundError` se assente)
2. Converte con `docling.document_converter.DocumentConverter` → markdown strutturato
3. Chunking con `RecursiveCharacterTextSplitter` (chunk_size=512, chunk_overlap=50)
4. Crea oggetti `Document` con `format_passage(chunk)` e metadata `{source, chunk_index}`
5. Chiama `faiss_store.add_documents(docs, filename)`

**Note implementative**:
- Tutti gli import pesanti (`langchain_text_splitters`, `docling`, `torch`) sono **lazy** (dentro la funzione) per evitare caricamento di torch al semplice import del modulo
- Estensioni supportate: `.pdf`, `.docx`, `.pptx`, `.html`, `.htm`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`

---

### `core/llm.py` — LLM locale GGUF

**Gestione DLL su Windows** (comportamento non pianificato originariamente, aggiunto in produzione):

La funzione `_register_windows_dll_dirs()` viene chiamata **prima** di qualsiasi `import llama_cpp` e registra le seguenti directory tramite `os.add_dll_directory`:
1. `src/ragchat/vendor/win_runtime/` — VC++ 2022 runtime bundled nel progetto
2. `<venv>/site-packages/llama_cpp/lib/` — DLL native di llama_cpp

L'ordine è critico: le VC++ runtime devono essere disponibili prima che il loader di Windows tenti di risolvere le dipendenze di `llama.dll`.

**Graceful degradation**: se `llama_cpp` non può essere importato (pacchetto assente o DLL mancante), `_LLAMA_AVAILABLE = False` e `is_llm_available()` restituisce `False` senza eccezioni. La GUI mostra un messaggio informativo anziché crashare.

**API pubblica**:
| Funzione | Comportamento |
|---|---|
| `is_llm_available()` | `True` se llama_cpp è importabile |
| `get_model_path()` | Percorso GGUF in cache HF (senza scaricare), o `None` |
| `get_llm()` | Singleton LlamaCpp; scarica il GGUF se non in cache |

---

### `core/rag.py` — RAGChain

**Pipeline** (`ask(question) → str`):
1. DB vuoto → messaggio informativo (no eccezione)
2. `format_query(question)` + `retriever.invoke()` → top-4 chunk
3. Costruzione contesto: chunk uniti da `"\n\n---\n\n"`
4. Prompt in italiano: "Rispondi basandoti esclusivamente sul contesto fornito..."
5. LLM non disponibile → messaggio informativo
6. `llm.invoke(prompt_text)` → risposta `.strip()`
7. Qualsiasi eccezione → log + messaggio di errore user-friendly (no crash)

**Note**: il LLM e gli import pesanti sono **lazy** (dentro `ask()`) per evitare import circolari e rallentamenti all'avvio.

---

### `ui/db_panel.py` — DBPanel

Widget `tk.Frame` per il pannello sinistro (30% della finestra).

**Layout** (dall'alto verso il basso):
- Label titolo "🗄 Database Vettoriale" (bold)
- Label percorso DB corrente (wrapping, colore muted `#57606a`)
- Pulsanti `Apri DB` / `Nuovo DB`
- Separatore
- Label contatore "Documenti (N):"
- `Listbox` documenti con scrollbar verticale
- Pulsanti `Aggiungi` / `Rimuovi sel.`
- Separatore
- Label di stato in basso (es. "● Pronto", "⏳ Indicizzazione...", "✔ 42 chunk indicizzati")

**Thread safety**: tutte le operazioni I/O/CPU girano in `threading.Thread(daemon=True)`; gli aggiornamenti GUI usano `self.after(0, callback)`. I pulsanti vengono disabilitati durante le operazioni e riabilitati al termine.

**Comportamenti**:
- `Apri DB`: `filedialog.askdirectory` → `FAISSStore.load(path)` in thread → `on_db_changed(store)`
- `Nuovo DB`: `filedialog.askdirectory` → `FAISSStore.create(path)` in thread → `on_db_changed(store)`
- `Aggiungi`: `filedialog.askopenfilenames` (filetypes da `get_supported_extensions()`) → `ingest_document()` per ogni file in thread; gestisce errori per file singolo senza interrompere il batch
- `Rimuovi sel.`: `messagebox.askyesno` → `store.remove_document(filename)` in thread

---

### `ui/chat_panel.py` — ChatPanel

Widget `tk.Frame` per il pannello destro (70% della finestra).

**Layout** (dall'alto verso il basso):
- Label titolo "💬 Chat" (bold)
- `ScrolledText` read-only (sfondo `#f5f5f5`, espandibile, scroll automatico in fondo)
  - Tag `"user"`: testo `#1a3a5c`, bold
  - Tag `"assistant"`: testo `#1a4a2a`
- Label stato "⏳ In elaborazione..." (nascosta di default, mostrata/nascosta con `grid()`/`grid_remove()`)
- Frame input: `ttk.Entry` + pulsante `Invia` (anche tasto `<Return>`)

**Thread safety**: la chiamata `rag_chain.ask()` gira in thread separato; il risultato torna alla GUI via `self.after(0, self._on_llm_done, answer)`. Input e pulsante vengono disabilitati durante l'elaborazione.

---

### `app.py` — MainApp

`MainApp(tk.Tk)` assembla i due pannelli:
- **Titolo**: "RAG Chat"
- **Dimensioni**: 1200×700 px, ridimensionabile, minsize 800×500
- **Tema ttk**: `vista` (Windows) → `aqua` (macOS) → `clam` (Linux/fallback)
- **Layout**: `ttk.PanedWindow(orient=HORIZONTAL)` con `weight=1` (DB) e `weight=3` (Chat)
- **Avvio**: tenta di caricare `~/faiss_db` silenziosamente se la cartella esiste
- **`_on_db_changed(store)`**: crea `RAGChain(store)`, aggiorna `ChatPanel` e `DBPanel`

---

### `__init__.py` — Entry point

```python
from ragchat.app import MainApp

def main() -> None:
    app = MainApp()
    app.mainloop()
```

Configurato in `pyproject.toml` come `ragchat = "ragchat:main"`.

---

## Differenze tra piano e implementazione reale

| Aspetto | Pianificato | Reale |
|---|---|---|
| **Python version** | `>=3.14` (poi abbassato a `>=3.12`) | `>=3.12,<3.13` |
| **Chunk IDs** | `list[int]` | `list[str]` (UUID assegnati da LangChain FAISS) |
| **`langchain.schema.Document`** | Usato nel piano | Sostituito con `langchain_core.documents.Document` (deprecazione LangChain) |
| **Import di `torch`/`langchain_text_splitters`** | Non considerato | Resi lazy in `ingestor.py` per evitare caricamento di torch all'avvio |
| **VC++ Redistributable** | Prerequisito di sistema (nota nel piano) | **Bundled nel progetto** in `vendor/win_runtime/`; nessun prerequisito di sistema |
| **`_register_windows_dll_dirs()`** | Non pianificato | Aggiunto in `llm.py` per caricare le DLL nell'ordine corretto su Windows |
| **`llama-cpp-python`** | Dipendenza standard | Wheel precompilato da GitHub releases in `[tool.uv.sources]`; marcato `sys_platform == 'win32'` |
| **`torch`** | Dipendenza standard | Wheel CPU-only da pytorch.org in `[tool.uv.sources]`; marcato `sys_platform == 'win32'` |
| **Avvio applicazione** | `uv run ragchat` | `uv run ragchat` ✓ |

---

## Flusso dati

```
Documento → Docling (markdown strutturato)
          → RecursiveCharacterTextSplitter (chunk 512/50)
          → format_passage("passage: " + chunk)
          → Document{page_content, metadata{source, chunk_index}}
          → FAISS.from_documents / add_documents
          → FAISSStore.save_local() + MetadataStore.save()

Query utente → format_query("query: " + question)
             → FAISSStore.as_retriever(k=4).invoke()
             → top-4 Document
             → Prompt IT: "Rispondi basandoti sul contesto..."
             → LlamaCpp.invoke(prompt)
             → Risposta → ChatPanel._append()
```

---

## Avvio

```bash
uv run ragchat
```

Al primo utilizzo vengono scaricati automaticamente:
- **multilingual-e5-large** (~560 MB) — al primo inserimento documento o query
- **gemma-2-2b-it-Q4_K_M.gguf** (~1.5 GB) — alla prima domanda in chat

Entrambi vengono messi in cache in `~/.cache/huggingface/hub/` e non riscaricati nelle sessioni successive.
