from __future__ import annotations

"""Version of RamaLamaPy."""


def version():
    return "0.18.0"


def print_version(args):
    if args.quiet:
        print(version())
    else:
        print("ramalama version %s" % version())


if __name__ == "__main__":
    print(version())
