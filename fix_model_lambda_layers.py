#!/usr/bin/env python3
"""
Script para corregir las capas Lambda problemáticas en el modelo entrenado
Reemplaza las capas Lambda sin output_shape con capas personalizadas equivalentes
"""

import tensorflow as tf
import numpy as np
from keras import models as kmodels
from keras import layers
import os
import sys

# Importar las capas personalizadas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.keras_custom_layers import (
    compute_mask_layer,
    squeeze_last_axis_func,
    mask_attention_scores_func,
    apply_attention_func,
    MaskedRepeatVector,
    AttentionLayer,
    func,
    SqueezeLastAxisLayer,
    ComputeMaskLayer,
    MaskAttentionScoresLayer,
    ApplyAttentionLayer,
)

def get_custom_objects():
    return {
        "compute_mask_layer": compute_mask_layer,
        "squeeze_last_axis_func": squeeze_last_axis_func,
        "mask_attention_scores_func": mask_attention_scores_func,
        "apply_attention_func": apply_attention_func,
        "MaskedRepeatVector": MaskedRepeatVector,
        "AttentionLayer": AttentionLayer,
        "func": func,
        "SqueezeLastAxisLayer": SqueezeLastAxisLayer,
        "ComputeMaskLayer": ComputeMaskLayer,
        "MaskAttentionScoresLayer": MaskAttentionScoresLayer,
        "ApplyAttentionLayer": ApplyAttentionLayer,
    }

def convert_lambda_to_custom_layer(layer, layer_name):
    """Convierte una capa Lambda problemática a una capa personalizada equivalente"""

    # Intentar identificar qué función Lambda está usando
    try:
        if hasattr(layer, 'function'):
            func_name = getattr(layer.function, '__name__', 'unknown')

            if 'squeeze' in func_name.lower() or 'squeeze_last_axis' in func_name.lower():
                print(f"Reemplazando Lambda {layer_name} con SqueezeLastAxisLayer")
                return SqueezeLastAxisLayer(name=layer_name)

            elif 'compute_mask' in func_name.lower() or 'mask' in func_name.lower():
                print(f"Reemplazando Lambda {layer_name} con ComputeMaskLayer")
                return ComputeMaskLayer(mask_value=0.0, name=layer_name)

            elif 'attention_scores' in func_name.lower():
                print(f"Reemplazando Lambda {layer_name} con MaskAttentionScoresLayer")
                return MaskAttentionScoresLayer(name=layer_name)

            elif 'apply_attention' in func_name.lower():
                print(f"Reemplazando Lambda {layer_name} con ApplyAttentionLayer")
                return ApplyAttentionLayer(name=layer_name)

    except Exception as e:
        print(f"Error analizando función Lambda {layer_name}: {e}")

    # Si no podemos identificar la función específica, usar un enfoque más genérico
    print(f"Usando SqueezeLastAxisLayer como fallback para {layer_name}")
    return SqueezeLastAxisLayer(name=layer_name)

def rebuild_model_with_custom_layers(original_model):
    """Reconstruye el modelo reemplazando las capas Lambda problemáticas"""

    # Obtener la configuración del modelo original
    config = original_model.get_config()

    # Procesar cada capa en la configuración
    new_layers = []

    for layer_config in config['layers']:
        layer_name = layer_config['name']
        layer_class = layer_config['class_name']

        if layer_class == 'Lambda':
            print(f"Encontrada capa Lambda problemática: {layer_name}")

            # Crear una configuración de capa personalizada basada en el contexto
            if 'squeeze' in layer_name.lower() or 'attention_score' in layer_name.lower():
                new_layer_config = {
                    'class_name': 'SqueezeLastAxisLayer',
                    'config': {'name': layer_name},
                    'name': layer_name,
                    'inbound_nodes': layer_config['inbound_nodes']
                }
            elif 'mask' in layer_name.lower() and 'compute' in layer_name.lower():
                new_layer_config = {
                    'class_name': 'ComputeMaskLayer',
                    'config': {'name': layer_name, 'mask_value': 0.0},
                    'name': layer_name,
                    'inbound_nodes': layer_config['inbound_nodes']
                }
            elif 'mask' in layer_name.lower() and 'attention' in layer_name.lower():
                new_layer_config = {
                    'class_name': 'MaskAttentionScoresLayer',
                    'config': {'name': layer_name},
                    'name': layer_name,
                    'inbound_nodes': layer_config['inbound_nodes']
                }
            elif 'apply' in layer_name.lower() and 'attention' in layer_name.lower():
                new_layer_config = {
                    'class_name': 'ApplyAttentionLayer',
                    'config': {'name': layer_name},
                    'name': layer_name,
                    'inbound_nodes': layer_config['inbound_nodes']
                }
            else:
                # Fallback genérico
                new_layer_config = {
                    'class_name': 'SqueezeLastAxisLayer',
                    'config': {'name': layer_name},
                    'name': layer_name,
                    'inbound_nodes': layer_config['inbound_nodes']
                }

            new_layers.append(new_layer_config)
            print(f"Reemplazado {layer_name} con {new_layer_config['class_name']}")
        else:
            # Mantener las demás capas como están
            new_layers.append(layer_config)

    # Actualizar la configuración con las nuevas capas
    config['layers'] = new_layers

    # Crear el nuevo modelo desde la configuración
    try:
        new_model = kmodels.Model.from_config(config, custom_objects=get_custom_objects())

        # Copiar los pesos desde el modelo original
        for orig_layer, new_layer in zip(original_model.layers, new_model.layers):
            if orig_layer.__class__.__name__ != 'Lambda':
                try:
                    new_layer.set_weights(orig_layer.get_weights())
                except Exception as e:
                    print(f"No se pudieron copiar pesos para {orig_layer.name}: {e}")

        return new_model

    except Exception as e:
        print(f"Error creando el nuevo modelo: {e}")
        return None

def main():
    model_path = "/Users/luis/DOCTORADO/Software/ARTIE_WEB/help-webservice/model/help_model.keras"
    backup_path = "/Users/luis/DOCTORADO/Software/ARTIE_WEB/help-webservice/model/help_model_backup.keras"
    fixed_path = "/Users/luis/DOCTORADO/Software/ARTIE_WEB/help-webservice/model/help_model_fixed.keras"

    print("Cargando modelo original...")

    try:
        # Cargar el modelo original con objetos personalizados
        original_model = kmodels.load_model(
            model_path,
            custom_objects=get_custom_objects(),
            compile=False,
            safe_mode=False
        )
        print("Modelo original cargado exitosamente")

        # Crear backup del modelo original
        print("Creando backup del modelo original...")
        original_model.save(backup_path)
        print(f"Backup guardado en: {backup_path}")

        # Reconstruir el modelo con capas personalizadas
        print("Reconstruyendo modelo con capas personalizadas...")
        fixed_model = rebuild_model_with_custom_layers(original_model)

        if fixed_model is not None:
            # Guardar el modelo corregido
            print("Guardando modelo corregido...")
            fixed_model.save(fixed_path)
            print(f"Modelo corregido guardado en: {fixed_path}")

            # Reemplazar el modelo original con el corregido
            print("Reemplazando modelo original con la versión corregida...")
            fixed_model.save(model_path)
            print("Modelo original actualizado exitosamente")

            print("\n¡Conversión completada!")
            print("El modelo ahora debería funcionar sin errores de Lambda layers")

        else:
            print("Error: No se pudo reconstruir el modelo")

    except Exception as e:
        print(f"Error durante la conversión: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
