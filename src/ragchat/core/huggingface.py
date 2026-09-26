"""Integrazione con HuggingFace Hub: lista modelli e gestione LLM locale.

La classe :class:`HuggingFace` centralizza tutto ciò che riguarda i modelli
HuggingFace nell'applicazione:

1. **Lista modelli** — scarica da HuggingFace API i modelli più scaricati per:
   - ``text-generation``: LLM per la chat (GGUF preferito, altrimenti safetensors)
   - ``feature-extraction``: modelli per gli embedding (safetensors)

   I modelli con più di 3 miliardi di parametri sono esclusi server-side
   tramite il filtro ``num_parameters="max:3B"`` di ``HfApi.list_models()``.

2. **Download modello** — scarica il file GGUF specifico tramite
   ``hf_hub_download``, oppure l'intero snapshot del repository tramite
   ``snapshot_download`` (per i modelli safetensors).

3. **Caricamento LLM** — istanzia il singleton LLM (``LlamaCpp`` per i file
   GGUF, ``HuggingFacePipeline`` per i repo safetensors) e lo mantiene in vita
   per la durata della sessione; lo invalida e ricarica automaticamente se il
   modello attivo cambia nella configurazione.

Ogni modello nella lista è un dizionario con le chiavi:

- ``id``: repo-id HuggingFace mostrato nella UI (es. ``"bartowski/gemma-2-2b-it-GGUF"``)
- ``url``: URL di download — punta al file GGUF specifico se disponibile,
  altrimenti alla pagina del repository
- ``format``: ``"gguf"`` oppure ``"safetensors"``
- ``tags``: lista di stringhe mostrate nel tooltip della listbox (include
  conteggio download, like e tag tecnici del repo)
"""

import logging
import re
from typing import Any, Callable, Optional, Tuple

import ragchat.vendor  # noqa: F401 – registra le DLL Windows all'avvio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Numero massimo di modelli da recuperare per categoria.
# Alzato a 500: con limit=500 e sort=downloads l'API risponde in ~0.5s
# restituendo comunque solo i modelli più rilevanti per l'utente.
_MAX_MODELS = 500

# Filtro parametri server-side: esclude modelli con più di 3 miliardi di parametri
_MAX_PARAMS_FILTER = "max:3B"

# Campi da espandere nella risposta di HfApi.list_models / model_info
_EXPAND_FIELDS = ["downloads", "likes", "tags", "siblings", "pipeline_tag"]

# Regex per URL HuggingFace a file specifico
# Formato: https://huggingface.co/{repo_id}/(blob|resolve)/{revision}/{filename}
_HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/([^/]+)/(.+)$"
)

# ---------------------------------------------------------------------------
# Verifica disponibilità llama-cpp-python (a livello di modulo)
# ---------------------------------------------------------------------------

try:
    import llama_cpp  # noqa: F401
    _LLAMA_AVAILABLE = True
    logger.debug("llama-cpp-python disponibile.")
except (ImportError, RuntimeError, OSError):
    _LLAMA_AVAILABLE = False
    logger.warning(
        "llama-cpp-python non e' installato o non caricabile. "
        "Il LLM locale non sara' disponibile. "
        "Per installarlo: pip install llama-cpp-python"
    )

# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------


class HuggingFace:
    """Gestisce la lista dei modelli HuggingFace e il ciclo di vita del LLM locale.

    Esempio::

        hf = HuggingFace()

        # Lista modelli da HuggingFace (max 3B parametri, server-side)
        models = hf.fetch_model_list(on_progress=lambda c, t: ...)

        # LLM locale (singleton lazy)
        if hf.is_llm_available():
            llm = hf.get_llm()
            answer = llm.invoke(prompt)
    """

    def __init__(self) -> None:
        """Inizializza la classe."""
        # Singleton LLM e spec (URL o repo-id) con cui è stato creato
        self._llm_instance = None
        self._loaded_spec: Optional[str] = None

    # ------------------------------------------------------------------
    # Disponibilità LLM
    # ------------------------------------------------------------------

    @staticmethod
    def is_llm_available() -> bool:
        """Restituisce True se llama-cpp-python oppure transformers è installato."""
        if _LLAMA_AVAILABLE:
            return True
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Spec del modello attivo (dalla configurazione)
    # ------------------------------------------------------------------

    @staticmethod
    def get_active_model_spec() -> str:
        """Legge lo spec (URL o repo-id) del modello LLM attivo dalla configurazione.

        Cerca nella lista ``query_models`` il dizionario il cui ``id`` corrisponde
        a ``query_model`` e ne restituisce il campo ``url``.  Questa stringa è
        usata internamente per distinguere i modelli GGUF (URL con path di file)
        dai modelli safetensors (repo-id semplice) e come chiave del singleton LLM.

        Returns:
            URL HuggingFace al file specifico (es. ``…/resolve/main/model.gguf``)
            oppure repo-id semplice (es. ``"owner/repo"``).
        """
        try:
            from ragchat.utils.config import Config, model_entry_id, model_entry_url
            cfg = Config.load()
            active_id = cfg["query_model"]
            for entry in cfg.get("query_models", []):
                if model_entry_id(entry) == active_id:
                    return model_entry_url(entry)
            return active_id
        except Exception:  # noqa: BLE001
            from ragchat.utils.config import Config
            first = Config.get_defaults()["query_models"][0]
            return first["url"]

    # ------------------------------------------------------------------
    # Utilità URL (statiche, solo uso interno)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_hf_url(url: str) -> Tuple[str, str]:
        """Estrae ``(repo_id, filename)`` da una URL HuggingFace.

        Raises:
            ValueError: se l'URL non è valida.
        """
        match = _HF_URL_RE.match(url)
        if not match:
            raise ValueError(f"URL HuggingFace non valida: {url}")
        return match.group(1), match.group(3)

    @staticmethod
    def _is_hf_url(spec: str) -> bool:
        """Restituisce True se *spec* è una URL HuggingFace a un file specifico."""
        return _HF_URL_RE.match(spec) is not None

    @staticmethod
    def _build_model_url(repo_id: str, filename: Optional[str] = None) -> str:
        """Costruisce l'URL canonica per un modello o un file su HuggingFace.

        Args:
            repo_id: identificativo del repository (es. ``"owner/repo"``).
            filename: nome del file all'interno del repo
                (es. ``"model-Q4_K_M.gguf"``).  Se fornito, l'URL punta al
                file tramite il branch ``main``; se ``None``, l'URL punta alla
                radice del repository.

        Returns:
            URL nella forma ``https://huggingface.co/{repo_id}/resolve/main/{filename}``
            oppure ``https://huggingface.co/{repo_id}``.
        """
        if filename:
            return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        return f"https://huggingface.co/{repo_id}"

    @staticmethod
    def _resolve_model_spec(spec: str) -> Tuple[str, Optional[str]]:
        """Risolve uno spec modello nella coppia ``(repo_id, filename)``.

        Args:
            spec: URL HuggingFace a un file specifico oppure repo-id semplice.

        Returns:
            ``(repo_id, filename)`` se *spec* è un'URL HuggingFace valida,
            oppure ``(spec, None)`` se *spec* è un repo-id semplice.
        """
        if HuggingFace._is_hf_url(spec):
            return HuggingFace._parse_hf_url(spec)
        return spec, None

    # ------------------------------------------------------------------
    # Cache locale
    # ------------------------------------------------------------------

    def get_cached_model_path(self) -> Optional[str]:
        """Restituisce il percorso locale del modello attivo se già presente in cache.

        Legge lo spec del modello attivo tramite :meth:`get_active_model_spec` e
        controlla la cache di HuggingFace Hub senza effettuare alcun download.

        Per i modelli GGUF (spec = URL con path di file) usa
        ``try_to_load_from_cache`` cercando il file specifico.
        Per i modelli safetensors (spec = repo-id) usa ``snapshot_download``
        con ``local_files_only=True``.

        Returns:
            Percorso assoluto al file GGUF in cache, o alla directory dello
            snapshot, se il modello è già stato scaricato; ``None`` altrimenti.
        """
        spec = self.get_active_model_spec()
        repo_id, filename = self._resolve_model_spec(spec)

        try:
            from huggingface_hub import try_to_load_from_cache, snapshot_download
        except ImportError:
            logger.warning("huggingface_hub non e' installato.")
            return None

        if filename is not None:
            cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
            if cached is not None and isinstance(cached, str):
                logger.debug("Modello trovato in cache: %s", cached)
                return cached
            return None

        try:
            path = snapshot_download(repo_id, local_files_only=True)
            logger.debug("Modello trovato in cache: %s", path)
            return path
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Download modello
    # ------------------------------------------------------------------

    def download_model(self, spec: Optional[str] = None) -> str:
        """Scarica il modello da HuggingFace Hub e restituisce il percorso locale.

        Se *spec* è un'URL HuggingFace con path di file, usa ``hf_hub_download``
        per scaricare quel solo file (tipicamente un ``.gguf``).
        Se *spec* è un repo-id semplice, usa ``snapshot_download`` per scaricare
        l'intero contenuto del repository.

        Args:
            spec: URL HuggingFace a un file specifico oppure repo-id semplice.
                Se ``None``, lo spec viene letto dalla configurazione tramite
                :meth:`get_active_model_spec`.

        Returns:
            Percorso assoluto al file scaricato (GGUF) o alla directory dello
            snapshot (repo-id).

        Raises:
            RuntimeError: se ``huggingface_hub`` non è installato o il download
                fallisce.
        """
        if spec is None:
            spec = self.get_active_model_spec()

        repo_id, filename = self._resolve_model_spec(spec)

        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub non e' installato. "
                "Installarlo con: pip install huggingface-hub"
            ) from exc

        if filename is not None:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            logger.info("Modello GGUF disponibile in: %s", path)
            return path

        logger.info("Download snapshot modello da: %s ...", repo_id)
        path = snapshot_download(repo_id)
        logger.info("Modello disponibile in: %s", path)
        return path

    # ------------------------------------------------------------------
    # Singleton LLM
    # ------------------------------------------------------------------

    def get_llm(self):
        """Restituisce il singleton LLM per il modello attivo (lazy, con auto-reload).

        Legge lo spec del modello attivo tramite :meth:`get_active_model_spec`.
        Se lo spec è cambiato rispetto all'ultima chiamata, il singleton
        precedente viene scartato e ne verrà creato uno nuovo.

        Al primo accesso (o dopo un cambio di modello):

        1. Controlla se il modello è già in cache locale
           (:meth:`get_cached_model_path`); se non lo è, lo scarica tramite
           :meth:`download_model`.
        2. Se lo spec è un'URL HuggingFace a un file (``.gguf``), istanzia
           ``LlamaCpp`` (richiede ``llama-cpp-python`` e ``langchain-community``).
           Se lo spec è un repo-id semplice, carica il modello con
           ``transformers`` tramite ``HuggingFacePipeline``
           (richiede ``transformers`` e ``langchain-community``).

        Returns:
            Istanza LLM già caricata (``LlamaCpp`` o ``HuggingFacePipeline``).

        Raises:
            ImportError: se ``llama-cpp-python``, ``transformers`` o
                ``langchain-community`` non sono installati.
            RuntimeError: se il download del modello fallisce.
        """
        spec = self.get_active_model_spec()

        if self._llm_instance is not None and self._loaded_spec != spec:
            logger.info(
                "Modello LLM cambiato (%s -> %s): singleton scartato.",
                self._loaded_spec, spec,
            )
            self._llm_instance = None
            self._loaded_spec = None

        if self._llm_instance is not None:
            return self._llm_instance

        model_path = self.get_cached_model_path()
        if model_path is None:
            model_path = self.download_model(spec)

        if self._is_hf_url(spec):
            self._llm_instance = self._create_llama_cpp(model_path)
        else:
            self._llm_instance = self._create_hf_llm(model_path)

        self._loaded_spec = spec
        logger.info("LLM '%s' pronto.", spec)
        return self._llm_instance

    # ------------------------------------------------------------------
    # Lista modelli da HuggingFace API
    # ------------------------------------------------------------------

    def fetch_model_list(
        self,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Scarica da HuggingFace la lista dei modelli disponibili.

        Vengono recuperati i modelli più scaricati (max 3B parametri) per:
        - ``text-generation`` (LLM) in formato GGUF o safetensors
        - ``feature-extraction`` (embedding) in formato safetensors

        Args:
            on_progress: callback opzionale ``(current, total)`` per la
                progress bar.  ``current`` va da 0 a ``total`` incluso.

        Returns:
            Dict con chiavi ``"query_models"`` e ``"embedding_models"``,
            ciascuna contenente una ``list[dict]`` con chiavi
            ``id``, ``url``, ``format``, ``tags``.

        Raises:
            RuntimeError: se la connessione a HuggingFace fallisce.
        """
        steps = [
            ("query",     "text-generation",  ["gguf", "safetensors"]),
            ("embedding", "feature-extraction", ["safetensors"]),
        ]
        total = len(steps)
        result: dict[str, list[dict[str, Any]]] = {
            "query_models": [],
            "embedding_models": [],
        }

        for i, (kind, task, formats) in enumerate(steps):
            if on_progress:
                on_progress(i, total)
            models = self._fetch_models_for_task(task, formats)
            result[f"{kind}_models"] = models
            logger.info("Recuperati %d modelli per '%s'.", len(models), task)

        if on_progress:
            on_progress(total, total)

        return result

    # ------------------------------------------------------------------
    # Internals — lista modelli
    # ------------------------------------------------------------------

    def _fetch_models_for_task(
        self,
        task: str,
        preferred_formats: list[str],
    ) -> list[dict[str, Any]]:
        """Recupera tramite ``HfApi`` i modelli per il pipeline tag *task*.

        Usa ``HfApi.list_models()`` con filtro server-side ``num_parameters``
        (max 3B) per escludere modelli troppo grandi.  Per ogni ``ModelInfo``
        restituito costruisce il dizionario con le chiavi ``id``, ``url``,
        ``format``, ``tags`` e ``pipeline_tag``.

        Args:
            task: pipeline tag HuggingFace (es. ``"text-generation"``).
            preferred_formats: formati da cercare in ordine di priorità
                (es. ``["gguf", "safetensors"]``).

        Returns:
            Lista di dizionari modello pronti per la configurazione, ordinata
            alfabeticamente per ``id``.
        """
        model_infos = self._api_search_models(task)
        result = []

        for info in model_infos:
            repo_id: str = info.id or ""
            if not repo_id:
                continue

            tags: list[str] = list(info.tags or [])
            downloads: int = info.downloads or 0
            likes: int = info.likes or 0
            pipeline_tag: str = info.pipeline_tag or task

            tooltip_tags = tags[:8]
            if downloads:
                tooltip_tags.insert(0, f"↓ {downloads:,} downloads")
            if likes:
                tooltip_tags.insert(1, f"♥ {likes:,} likes")

            model_format, model_url = self._resolve_format_and_url(
                repo_id, info.siblings or [], tags, preferred_formats
            )
            if model_format is None:
                model_format = "safetensors"
                model_url = HuggingFace._build_model_url(repo_id)

            result.append({
                "id": repo_id,
                "url": model_url,
                "format": model_format,
                "tags": tooltip_tags,
                "pipeline_tag": pipeline_tag,
            })

        result.sort(key=lambda m: m["id"].lower())
        return result

    @staticmethod
    def _resolve_format_and_url(
        repo_id: str,
        siblings: list,
        tags: list[str],
        preferred_formats: list[str],
    ) -> tuple[Optional[str], str]:
        """Determina il formato e l'URL di download ottimali per un modello.

        Scorre *preferred_formats* in ordine e restituisce al primo formato
        trovato:

        - ``"gguf"``: cerca tra i ``siblings`` un file ``.gguf``, preferendo
          quello con ``Q4_K_M`` nel nome (buon compromesso qualità/dimensione);
          in assenza di siblings con ``.gguf``, si basa sul tag ``"gguf"``.
        - ``"safetensors"``: verifica la presenza di un file ``.safetensors``
          tra i siblings o del tag ``"safetensors"``.

        Se nessun formato preferito è rilevabile restituisce ``(None, url_repo)``.

        Args:
            repo_id: identificativo del repository (es. ``"owner/repo"``).
            siblings: lista di oggetti ``RepoSibling`` (o dict con ``rfilename``).
            tags: lista di tag stringa del repository.
            preferred_formats: formati da cercare in ordine di priorità.

        Returns:
            Tupla ``(formato, url)`` dove *formato* è ``"gguf"``,
            ``"safetensors"`` oppure ``None`` se non determinabile.
        """
        def _rfilename(s) -> str:
            """Estrae il nome file sia da RepoSibling che da dict."""
            if hasattr(s, "rfilename"):
                return s.rfilename or ""
            return s.get("rfilename", "") if isinstance(s, dict) else ""

        lower_tags = [t.lower() for t in tags]

        for fmt in preferred_formats:
            if fmt == "gguf":
                gguf_files = [
                    _rfilename(s)
                    for s in siblings
                    if _rfilename(s).lower().endswith(".gguf")
                ]
                if gguf_files:
                    chosen = next(
                        (f for f in gguf_files if "Q4_K_M" in f), gguf_files[0]
                    )
                    return "gguf", HuggingFace._build_model_url(repo_id, chosen)
                if "gguf" in lower_tags:
                    return "gguf", HuggingFace._build_model_url(repo_id)

            elif fmt == "safetensors":
                has_st = any(
                    _rfilename(s).lower().endswith(".safetensors")
                    for s in siblings
                ) or "safetensors" in lower_tags
                if has_st:
                    return "safetensors", HuggingFace._build_model_url(repo_id)

        return None, HuggingFace._build_model_url(repo_id)

    def fetch_model_info(self, repo_id: str) -> dict[str, Any]:
        """Recupera da HuggingFace i metadati di un singolo repository e restituisce
        un dizionario modello pronto all'uso (stesso formato di :meth:`fetch_model_list`).

        Args:
            repo_id: identificativo del repository, es. ``"bartowski/gemma-2-2b-it-GGUF"``.
                Può contenere anche un'URL HuggingFace completa: in tal caso viene
                estratto automaticamente il ``repo_id``.

        Returns:
            Dizionario con chiavi ``id``, ``url``, ``format``, ``tags``,
            ``pipeline_tag``.

        Raises:
            RuntimeError: se il repository non esiste o la connessione fallisce.
        """
        # Normalizza: se l'utente incolla un'URL estrae il repo_id
        if self._is_hf_url(repo_id):
            repo_id, _ = self._parse_hf_url(repo_id)

        logger.debug("Richiesta info modello HuggingFace: %s", repo_id)
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.model_info(repo_id, expand=_EXPAND_FIELDS)
        except Exception as exc:
            raise RuntimeError(
                f"Impossibile recuperare il modello '{repo_id}' da HuggingFace: {exc}"
            ) from exc

        # Determina pipeline_tag
        pipeline_tag: str = info.pipeline_tag or ""

        # Scegli i formati preferiti in base al task
        if pipeline_tag == "feature-extraction":
            preferred_formats = ["safetensors"]
        else:
            preferred_formats = ["gguf", "safetensors"]

        tags: list[str] = list(info.tags or [])
        model_format, model_url = self._resolve_format_and_url(
            repo_id, info.siblings or [], tags, preferred_formats
        )
        if model_format is None:
            model_format = "safetensors"
            model_url = self._build_model_url(repo_id)

        # Costruisce i tag tooltip
        downloads: int = info.downloads or 0
        likes: int = info.likes or 0
        tooltip_tags = tags[:8]
        if downloads:
            tooltip_tags.insert(0, f"↓ {downloads:,} downloads")
        if likes:
            tooltip_tags.insert(1, f"♥ {likes:,} likes")

        return {
            "id": repo_id,
            "url": model_url,
            "format": model_format,
            "tags": tooltip_tags,
            "pipeline_tag": pipeline_tag,
        }

    def _api_search_models(self, task: str):
        """Chiama ``HfApi.list_models()`` e restituisce i modelli per *task*.

        Usa il filtro server-side ``num_parameters="max:3B"`` per escludere
        modelli troppo grandi, ordina per download decrescente e recupera al
        massimo ``_MAX_MODELS`` risultati.

        Args:
            task: pipeline tag HuggingFace (es. ``"text-generation"``).

        Returns:
            Iterabile di oggetti ``ModelInfo``.

        Raises:
            RuntimeError: se la connessione a HuggingFace fallisce.
        """
        logger.debug("Ricerca modelli HuggingFace: task=%s", task)
        try:
            from ragchat.utils.config import Config
            max_models = Config.load().get("hf_max_models", _MAX_MODELS)
        except Exception:  # noqa: BLE001
            max_models = _MAX_MODELS
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            return list(
                api.list_models(
                    pipeline_tag=task,
                    sort="downloads",
                    num_parameters=_MAX_PARAMS_FILTER,
                    limit=max_models,
                    expand=_EXPAND_FIELDS,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Impossibile contattare HuggingFace API (task={task}): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internals — creazione LLM
    # ------------------------------------------------------------------

    @staticmethod
    def _create_llama_cpp(model_path: str):
        """Istanzia e restituisce un oggetto ``LlamaCpp`` da un file GGUF locale.

        Args:
            model_path: percorso assoluto al file ``.gguf`` già scaricato.

        Returns:
            Istanza ``langchain_community.llms.LlamaCpp`` pronta per l'uso.

        Raises:
            ImportError: se ``llama-cpp-python`` o ``langchain-community``
                non sono installati.
        """
        if not _LLAMA_AVAILABLE:
            raise ImportError(
                "llama-cpp-python non e' installato. "
                "Installarlo con: pip install llama-cpp-python"
            )
        try:
            from langchain_community.llms import LlamaCpp
        except ImportError as exc:
            raise ImportError(
                "langchain-community non e' installato. "
                "Installarlo con: pip install langchain-community"
            ) from exc

        logger.info("Caricamento modello LLM GGUF da: %s", model_path)
        return LlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            temperature=0.7,
            max_tokens=512,
            n_threads=4,
            verbose=False,
        )

    @staticmethod
    def _create_hf_llm(repo_path: str):
        """Istanzia un LLM HuggingFace da una directory snapshot locale.

        Carica tokenizer e modello tramite ``transformers``, crea una pipeline
        ``text-generation`` e la avvolge in ``HuggingFacePipeline`` di
        ``langchain-community``.

        Args:
            repo_path: percorso assoluto alla directory snapshot del repository
                (restituita da ``snapshot_download``).

        Returns:
            Istanza ``langchain_community.llms.HuggingFacePipeline`` pronta
            per l'uso.

        Raises:
            ImportError: se ``transformers`` o ``langchain-community``
                non sono installati.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers non e' installato. "
                "Installarlo con: pip install transformers"
            ) from exc
        try:
            from langchain_community.llms import HuggingFacePipeline
        except ImportError as exc:
            raise ImportError(
                "langchain-community non e' installato. "
                "Installarlo con: pip install langchain-community"
            ) from exc

        logger.info("Caricamento modello LLM transformers da: %s", repo_path)
        tokenizer = AutoTokenizer.from_pretrained(repo_path)
        model = AutoModelForCausalLM.from_pretrained(repo_path)
        hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
        return HuggingFacePipeline(pipeline=hf_pipe)
