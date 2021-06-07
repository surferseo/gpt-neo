from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast
from tensorflow.python.tpu import tpu_config, tpu_estimator
import tensorflow.compat.v1 as tf
from model_fns import model_fn
from functools import partial

params = {'colab': True}

def get_estimator(path, params):

    if params['colab']:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    else:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver('colab')

    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=path,
        save_checkpoints_steps=None,  # Disable the default saver
        save_checkpoints_secs=None,  # Disable the default saver
        tpu_config=tpu_config.TPUConfig(
            per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))

    return tpu_estimator.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=config,
        train_batch_size=1,
        eval_batch_size=1,
        predict_batch_size=1,
        params=params)

def input_fn(params, texts, tokenizer):
    input_ids = tokenizer(
        texts,
        return_tensors="tf",
        max_length=2048,
        truncation=True,
        padding=True,
    )["input_ids"]

    dataset = tf.data.Dataset.from_tensors(input_ids)

    def _dummy_labels(x):
        return x, tf.constant(0)

    dataset = dataset.map(_dummy_labels)
    return dataset

def predict(texts, estimator):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
    input_fn = partial(input_fn, texts=texts, tokenizer=tokenizer)
    predictions = estimator.predict(input_fn=input_fn)
    for i, p in enumerate(predictions):
        p = p["outputs"]
        text = tokenizer.batch_decode(
            p, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(f'OUTPUT:\n\n{text}\n\n')