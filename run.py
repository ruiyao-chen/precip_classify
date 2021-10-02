from atmos.utils.context import Context
from atmos.model import build_runner


if __name__ == "__main__":
    ctx = Context()
    runner = build_runner(ctx)
    runner.submit()
