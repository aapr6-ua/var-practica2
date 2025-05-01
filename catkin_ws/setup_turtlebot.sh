#!/bin/bash

# Detectar automáticamente la carpeta actual donde está el script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configurar el modelo de Turtlebot3
export TURTLEBOT3_MODEL=waffle

# Source del devel/setup.bash relativo al directorio detectado
source "$SCRIPT_DIR/devel/setup.bash"

echo "Configurado TURTLEBOT3_MODEL=waffle y cargado workspace en: $SCRIPT_DIR"
