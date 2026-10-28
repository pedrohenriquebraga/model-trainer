import os
import sys
import argparse
from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# import shutil
# import tempfile


def find_latest_model(models_dir, extension):
    if not os.path.isdir(models_dir):
        return None

    candidates = [
        os.path.join(models_dir, file_name)
        for file_name in os.listdir(models_dir)
        if file_name.endswith(f".{extension}")
    ]
    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def resolve_input_model_path(input_model_path):
    if os.path.exists(input_model_path):
        return input_model_path

    latest_keras = find_latest_model('models', 'keras')
    if latest_keras is not None:
        print(f"Aviso: {input_model_path} nao encontrado. Usando ultimo .keras: {latest_keras}")
        return latest_keras

    raise FileNotFoundError(f"Modelo nao encontrado: {input_model_path}")


def resolve_output_model_path(output_model_path, input_model_path, quantization_mode, runtime_mode):
    output_dir = os.path.dirname(output_model_path) or 'models'
    os.makedirs(output_dir, exist_ok=True)

    if output_model_path.endswith('efficient_det.tflite'):
        input_stem = os.path.splitext(os.path.basename(input_model_path))[0]
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        base_name = f"{input_stem}_{quantization_mode}_{runtime_mode}_{run_id}"
        candidate = os.path.join(output_dir, f"{base_name}.tflite")
    else:
        candidate = output_model_path

    if not os.path.exists(candidate):
        return candidate

    stem, ext = os.path.splitext(candidate)
    idx = 1
    while True:
        indexed_candidate = f"{stem}_{idx}{ext}"
        if not os.path.exists(indexed_candidate):
            return indexed_candidate
        idx += 1

def detect_model_input_shape(model):
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape is None or len(input_shape) != 4:
        raise ValueError(f"Shape de entrada invalido: {input_shape}")

    _, height, width, channels = input_shape

    if height is None or width is None:
        raise ValueError(
            "Nao foi possivel detectar resolucao fixa do modelo (height/width indefinidos)."
        )

    if channels is None:
        channels = 3

    return int(height), int(width), int(channels)


def representative_dataset(img_size):
    base_dir = './data'
    train_dir = os.path.join(base_dir, 'train')
    batch_size = 4
    
    if not os.path.exists(train_dir):
        print(f"Aviso: diretório {train_dir} não encontrado. Usando dataset vazio para quantização.")
        return
    
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size)
    
    def preprocess(images, labels):
        return tf.cast(images, tf.float32), labels
    
    train_dataset = train_dataset.map(preprocess)
    
    for images, _ in train_dataset.unbatch().take(150):
        yield [tf.expand_dims(tf.cast(images, tf.float32), axis=0)]




def convert_keras_to_tflite(input_model_path, output_model_path, quantization_mode, runtime_mode):    
    input_model_path = resolve_input_model_path(input_model_path)
    output_model_path = resolve_output_model_path(output_model_path, input_model_path, quantization_mode, runtime_mode)
    
    # 1. Força a política de precisão global em float32 pura
    tf.keras.mixed_precision.set_global_policy('float32')
    
    print(f"Carregando modelo original de {input_model_path}...")
    trained_model = tf.keras.models.load_model(input_model_path, compile=False)
    input_height, input_width, input_channels = detect_model_input_shape(trained_model)
    num_classes = trained_model.output_shape[-1]
    print(f"Propriedades: ({input_height}x{input_width}x{input_channels}) | Classes: {num_classes}")
    
    # O modelo salvo contém o backbone achatado no Functional principal.
    # Clone o grafo inteiro para remover políticas mixed-precision sem perder
    # conexões internas do ConvNeXt.
    def clone_layer(layer):
        config = layer.get_config()
        dtype_config = config.get("dtype")
        if isinstance(dtype_config, dict):
            dtype_config.setdefault("config", {})["name"] = "float32"
        elif dtype_config is not None:
            config["dtype"] = "float32"
        return layer.__class__.from_config(config)

    clean_model = tf.keras.models.clone_model(
        trained_model,
        clone_function=clone_layer,
    )
    clean_model.set_weights([
        np.asarray(weight, dtype=np.float32)
        for weight in trained_model.get_weights()
    ])
    print("✓ Grafo clonado e convertido para float32 puro.")

    # 3. Conversão TFLite
    preferred_modes = [quantization_mode]
    if quantization_mode in ['int8', 'float16']:
        preferred_modes.extend(['dynamic', 'float'])
    
    runtime_candidates = ['builtin'] if runtime_mode == 'builtin' else (['flex'] if runtime_mode == 'flex' else ['builtin', 'flex'])

    tflite_model = None
    selected_mode = None
    selected_runtime = None
    last_error = None

    for runtime_candidate in runtime_candidates:
        for mode in preferred_modes:
            try:
                print(f"Tentando conversão com modo '{mode}' e runtime '{runtime_candidate}'...")

                concrete_function = tf.function(clean_model).get_concrete_function(
                    tf.TensorSpec(
                        [1, input_height, input_width, input_channels],
                        dtype=tf.float32,
                    )
                )
                frozen_function = convert_variables_to_constants_v2(concrete_function)
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [frozen_function],
                    clean_model,
                )
                converter.experimental_enable_resource_variables = False
                converter.allow_custom_ops = False

                if runtime_candidate == 'builtin':
                    if mode == 'int8':
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    else:
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                else:
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]

                if mode == 'dynamic':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                elif mode == 'int8':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = lambda: representative_dataset((input_height, input_width))
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                elif mode == 'float16':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]

                print("Gerando os bytes do TFLite nativo...")
                tflite_model = converter.convert()
                selected_mode = mode
                selected_runtime = runtime_candidate
                print(f"✓ Conversão bem-sucedida com modo '{mode}' e runtime '{runtime_candidate}'")
                break
            except Exception as conversion_error:
                last_error = conversion_error
                print(f"⚠ Falha na conversão com modo '{mode}' e runtime '{runtime_candidate}': {conversion_error}")

        if tflite_model is not None:
            break
    
    if tflite_model is None:
        raise RuntimeError(f"Nenhuma conversão TFLite teve sucesso. Último erro: {last_error}")
    
    # Salva o arquivo TFLite
    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)
    with open(output_model_path, "wb") as f:
        f.write(tflite_model)
    
    file_size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"\n✓ Modelo convertido com sucesso!")
    print(f"  Arquivo: {output_model_path}")
    print(f"  Tamanho: {file_size_mb:.2f} MB")
    print(f"  Quantização: {selected_mode}")
    print(f"  Runtime TFLite: {selected_runtime}")
    
def main():
    parser = argparse.ArgumentParser(
        description="Converter modelo Keras para TFLite"
    )
    parser.add_argument(
        '--input',
        default='models/efficient_det.keras',
        help='Caminho do modelo .keras (padrão: models/efficient_det.keras)'
    )
    parser.add_argument(
        '--output',
        default='models/efficient_det.tflite',
        help='Caminho do modelo .tflite (padrão: models/efficient_det.tflite)'
    )
    parser.add_argument(
        '--quantization',
        choices=['int8', 'float16', 'dynamic', 'float'],
        default='dynamic',
        help='Modo de quantização (padrão: dynamic). Para ConvNeXt, dynamic ou float são mais estáveis.'
    )
    parser.add_argument(
        '--runtime',
        choices=['builtin', 'flex', 'auto'],
        default='builtin',
        help="Runtime TFLite alvo (padrão: builtin, sem Flex). 'auto' tenta builtin e depois flex."
    )
    args = parser.parse_args()
    
    try:
        convert_keras_to_tflite(args.input, args.output, args.quantization, args.runtime)
    except Exception as e:
        print(f"\n✗ Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()