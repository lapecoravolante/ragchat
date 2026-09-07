from ragchat.app import MainApp
import logging

def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Avvio applicazione RAG Chat")


    app = MainApp()
    app.mainloop()
