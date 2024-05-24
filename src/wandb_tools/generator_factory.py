import garch_index_generator


def generator_factory(model: str):
    if model == "univar_garch":
        return garch_index_generator.GarchIndexGenerator()

    else:
        raise ValueError(f"{model} could not be converted to a generator!")
