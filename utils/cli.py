import argparse




def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project_root",
        default="/Users/darren/资料",
        type=str,
        help="The project root to folder SPIF_DU, e.g. /example/works/du"
    )

    parser.add_argument(
        "--grid",
        default=15,
        type=int,
        help="Training with which grid size. Availiable options: (5,10,15,20)"
    )

    parser.add_argument(
        "--load_model",
        default="",
        type=str,
        help="Name of the model to be evaluated in trained_models folder."
    )

    return parser