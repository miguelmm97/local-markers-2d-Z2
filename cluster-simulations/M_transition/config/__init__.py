import pathlib
import tomli

path = pathlib.Path(__file__).parent / "parameters_M_transition.toml"
with path.open(mode="rb") as fp:
    parameters_M_transition = tomli.load(fp)
