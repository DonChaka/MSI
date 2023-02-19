from logicGateNetwork import Layer, LogicGateNetwork
from geneticEngine import GeneticEngine
import numpy as np


def evaluate_config(config: str, x: np.ndarray, y: np.ndarray, network: LogicGateNetwork) -> float:
    network.set_params(config)
    y_pred = network.forward(x)
    return sum(np.logical_xor(y, y_pred))


data = np.random.randint(2, size=(100, 10))

target_model = LogicGateNetwork([
    Layer(n_features=data.shape[1], units=8),
    Layer(n_features=8, units=4),
    Layer(n_features=4, units=1)
])

target_pred = target_model.forward(data)

training_model = LogicGateNetwork([
    Layer(n_features=data.shape[1], units=8),
    Layer(n_features=8, units=4),
    Layer(n_features=4, units=1)
])
conf_len = training_model.get_feature_vector_length()

engine = GeneticEngine(
    evaluate_config,
    n_features=1,
    feature_size=conf_len,
    n_cut_points=int(conf_len / 22),
    n_mutate_points=int(conf_len / 22),
    cost_function_args=(data, target_pred, training_model)
)

engine.run(n_generations=100, verbose=True)

print(evaluate_config(engine.get_best()[0], data, target_pred, training_model))
