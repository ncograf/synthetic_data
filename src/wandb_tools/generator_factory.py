import garch_copula_index_generator
import garch_index_generator


def generator_factory(model: str):
    if model == "univar_garch":
        return garch_index_generator.GarchIndexGenerator()
    if model == "copula_garch":
        return garch_copula_index_generator.GarchCopulaIndexGenerator()
    else:
        raise ValueError(f"{model} could not be converted to a generator!")
