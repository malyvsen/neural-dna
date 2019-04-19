class Weights:
    def __init__(self, creator, num_units):
        self.num_units = num_units
        self.creator = creator



class Biases:
    def __init__(self, creator):
        self.creator = creator



class Activation:
    def __init__(self, function):
        self.function = function



def generate(transformations, weights_creator, biases_creator):
    result = []
    for transformation in transformations:
        if isinstance(transformation, int):
            result.append(Weights(weights_creator, transformation))
            result.append(Biases(biases_creator))
        elif callable(transformation):
            result.append(Activation(function=transformation))
        else:
            raise TypeError('unrecognized transformation type')
    return result
