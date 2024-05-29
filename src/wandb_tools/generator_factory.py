import fourier_flow_index_generator
import garch_copula_index_generator
import garch_index_generator


def generator_factory(model: str):
    if model == "univar_garch":
        return garch_index_generator.GarchIndexGenerator()
    if model == "copula_garch":
        return garch_copula_index_generator.GarchCopulaIndexGenerator()
    if model == "fourier_flow":
        return fourier_flow_index_generator.FourierFlowIndexGenerator()
    else:
        raise ValueError(f"{model} could not be converted to a generator!")
