import sys
import importlib


def get_version(pkg_name):
    try:
        pkg = importlib.import_module(pkg_name)
        return getattr(pkg, "__version__", "<no __version__>")
    except Exception as e:
        return f"not installed ({e})"


def main():
    print("Python executable:", sys.executable)
    for pkg in ("numpy", "pandas", "sklearn"):
        print(f"{pkg} version:", get_version(pkg))


if __name__ == "__main__":
    main()
