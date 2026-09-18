import pathlib
import tomli

path = pathlib.Path(__file__).parent / "parameters_W_transition.toml"
with path.open(mode="rb") as fp:
    parameters_W_transition = tomli.load(fp)