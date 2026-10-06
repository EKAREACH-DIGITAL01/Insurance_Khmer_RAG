"""Streamlit entry point. Run with: streamlit run app.py."""


def main() -> None:
    from khmer_rag.bootstrap import configure_runtime

    configure_runtime()

    from khmer_rag.ui.application import main as run_application

    run_application()


if __name__ == "__main__":
    main()
