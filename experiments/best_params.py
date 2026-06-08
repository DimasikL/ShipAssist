def get_model_best_params(model_name: str) -> dict:
    if model_name == 'whisper':
        # tiny 0.95 tests no-fuzzy
        return {
            'overlapping': 0.7893139756656726, 'block_size': 1.7196965370864798, 'prep_name': 'no',
            'model_args': {
                'model_mode': 'tiny', 'fuzzy_th': 68.5139657451007,
                'prompt_key': 'контекст управления',
                'transcribe_kwargs': {
                    'best_of': 7, 'beam_size': 6,
                    'compression_ratio_threshold': 3.1935429997159086,
                    'logprob_threshold': -0.5167847932566585,
                    'no_speech_threshold': 0.3456709736680613
                }
            }
        }
    elif model_name == 'gigaam':
        return {
            'overlapping': 0.7893139756656726, 'block_size': 1.7196965370864798, 'prep_name': 'no',
            'model_args': {
                'model_mode': 'rnnt', 'fuzzy_th': 68.5139657451007,
                'prompt_key': 'контекст управления'
            }
        }
    elif model_name == 'america':
        return {
            'overlapping': None, 'block_size': None,
            'prep_name': None,
            'model_args': {}
        }
    elif model_name == 'w2v2':
        return {
            'overlapping': 0.7893139756656726, 'block_size': 1.7196965370864798, 'prep_name': 'no',
            'model_args': {
                'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
                'fuzzy_th': 68.
            }
        }
    elif model_name == 'wtv_clf':
        return {
            'overlapping': 0.7893139756656726, 'block_size': 1.7196965370864798, 'prep_name': 'no',
            'model_args': {
                'th': 0.8
            }
        }
        # return {
        #     'overlapping': 0.45, 'block_size': 3.430481540976544, 'prep_name': '3', 'reset_if_found': True,
        #     'model_args': {'th': 0.5787253895194393}
        # }
    elif model_name == 'ft w2v2':
        return {
            'overlapping': 0.5,
            'block_size': 1.5,
            'prep_name': 'no',
            'model_args': {
                'model_path': r'C:\Users\Dmitriy\PycharmProjects\ShipAssistant\full_tune\models\run_2025-10-08_20-44-04\best_model',
                'th': 0.8
            }
        }

    else:
        raise ValueError(f"Undefined model_name = {model_name}")
