import os


ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

ZOO_ROOT = os.path.join(ROOT, "zoo_new")  # TODO: change back to zoo when everything done

SAVE_ROOT = os.path.join(os.getcwd(), "output")


__all__ = [
    "ROOT",
    "ZOO_ROOT",
    "SAVE_ROOT",
]
