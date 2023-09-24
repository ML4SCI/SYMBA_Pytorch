from models import seq2seq_transformer, BART, LED

registered_models = [seq2seq_transformer.Model, BART.Model, LED.Model]

def get_model_from_config(config):
    model_name = config.model_name
    for register_model in registered_models:
        if register_model.is_valid_model_name(model_name):
            return register_model.get_model_from_config(config)
    raise ValueError(f"Model {model_name} is not registered.")
