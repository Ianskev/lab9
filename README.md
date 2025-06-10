# Análisis de Fragmentación en PostgreSQL

Este proyecto realiza un análisis comparativo entre diferentes estrategias de fragmentación (particionamiento) en PostgreSQL, evaluando el rendimiento de consultas con tablas no fragmentadas, fragmentación anidada y fragmentación compuesta.

## Requisitos

- Python 3.6+
- PostgreSQL 11+ (con soporte para particionamiento declarativo)
- Bibliotecas de Python:
  - psycopg2
  - pandas
  - matplotlib
  - numpy
  - python-dotenv

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Configuración

### Archivo `.env`

Crea un archivo `.env` en el directorio principal con la siguiente información:

```properties
DB_HOST=localhost
DB_NAME=lab9
DB_USER=postgres
DB_PASSWORD=tu_contraseña

DATA_DIR=/ruta/completa/a/la/carpeta/data2
```
