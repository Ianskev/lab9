import psycopg2
import time
import os
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "db": {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME", "lab9"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres")
    },
    "data_dir": os.getenv("DATA_DIR", "/data2")
}

RESULTS_DIR = os.path.join(os.getcwd(), "resultados")

QUERIES = {
    "Query 1": {
        "description": "Empleados contratados en 1990 con salario > 50000",
        "regular": """
            SELECT * FROM employees 
            WHERE date_part('year', hire_date) = 1990 AND salary > 50000;
        """,
        "nested": """
            SELECT * FROM employees3 
            WHERE date_part('year', hire_date) = 1990 AND salary > 50000;
        """,
        "composite": """
            SELECT * FROM employees3_composite 
            WHERE date_part('year', hire_date) = 1990 AND salary > 50000;
        """
    },
    "Query 2": {
        "description": "Promedio salarial por año de contratación",
        "regular": """
            SELECT date_part('year', hire_date) as year, AVG(salary) as avg_salary
            FROM employees 
            GROUP BY date_part('year', hire_date)
            ORDER BY year;
        """,
        "nested": """
            SELECT date_part('year', hire_date) as year, AVG(salary) as avg_salary
            FROM employees3 
            GROUP BY date_part('year', hire_date)
            ORDER BY year;
        """,
        "composite": """
            SELECT date_part('year', hire_date) as year, AVG(salary) as avg_salary
            FROM employees3_composite 
            GROUP BY date_part('year', hire_date)
            ORDER BY year;
        """
    },
    "Query 3": {
        "description": "Distribución salarial por período",
        "regular": """
            SELECT 
                date_part('year', hire_date) as year,
                CASE 
                    WHEN salary < 40000 THEN 'Bajo'
                    WHEN salary BETWEEN 40000 AND 60000 THEN 'Medio'
                    ELSE 'Alto'
                END as rango_salarial,
                COUNT(*) as cantidad
            FROM employees
            GROUP BY year, rango_salarial
            ORDER BY year, rango_salarial;
        """,
        "nested": """
            SELECT 
                date_part('year', hire_date) as year,
                CASE 
                    WHEN salary < 40000 THEN 'Bajo'
                    WHEN salary BETWEEN 40000 AND 60000 THEN 'Medio'
                    ELSE 'Alto'
                END as rango_salarial,
                COUNT(*) as cantidad
            FROM employees3
            GROUP BY year, rango_salarial
            ORDER BY year, rango_salarial;
        """,
        "composite": """
            SELECT 
                date_part('year', hire_date) as year,
                CASE 
                    WHEN salary < 40000 THEN 'Bajo'
                    WHEN salary BETWEEN 40000 AND 60000 THEN 'Medio'
                    ELSE 'Alto'
                END as rango_salarial,
                COUNT(*) as cantidad
            FROM employees3_composite
            GROUP BY year, rango_salarial
            ORDER BY year, rango_salarial;
        """
    }
}

def get_table_type(variant_name):
    return {
        "regular": "Sin Fragmentación",
        "nested": "Fragmentación Anidada", 
        "composite": "Fragmentación Compuesta"
    }.get(variant_name, "Desconocido")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Creado directorio: {directory}")

def connect_to_postgres():
    try:
        conn = psycopg2.connect(**CONFIG["db"])
        print("Conexión exitosa a PostgreSQL")
        return conn
    except Exception as e:
        print(f"Error al conectar a PostgreSQL: {e}")
        exit(1)

def load_csv_to_table(cursor, csv_path, table, date_columns):
    print(f"Leyendo archivo CSV: {csv_path}")
    data = pd.read_csv(csv_path)
    
    for col in date_columns:
        if col in data.columns:
            data[col] = data[col].replace('9999-01-01', '2999-12-31')
            data[col] = pd.to_datetime(data[col], errors='coerce')
    
    buffer = StringIO()
    data.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    
    cursor.copy_from(buffer, table, sep=',', null='')
    print(f"Cargados {len(data)} registros en {table}")
    return len(data)

def extract_and_load_data(conn):
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS employees CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS salaries CASCADE;")
    
    cursor.execute("""
    CREATE TABLE employees (
      emp_no int,
      birth_date date,
      first_name varchar(14),
      last_name varchar(16),
      gender character(1),
      hire_date date,
      dept_no varchar(5),
      from_date date
    );
    """)
    
    cursor.execute("""
    CREATE TABLE salaries (
      emp_no int,
      salary int,
      from_date date,
      to_date date 
    );
    """)

    # Construir rutas completas a partir del directorio de datos
    employees_csv_path = os.path.join(CONFIG["data_dir"], "employees.csv")
    salaries_csv_path = os.path.join(CONFIG["data_dir"], "salaries.csv")

    load_csv_to_table(cursor, employees_csv_path, "employees", 
                     ['birth_date', 'hire_date', 'from_date'])
    
    load_csv_to_table(cursor, salaries_csv_path, "salaries", 
                     ['from_date', 'to_date'])
    
    conn.commit()
    print("Datos cargados correctamente")

def add_salary_column(conn):
    cursor = conn.cursor()
    
    print("  - Añadiendo columna salary a la tabla employees...")
    cursor.execute("ALTER TABLE employees ADD COLUMN IF NOT EXISTS salary INTEGER;")
    print("  - Columna añadida correctamente.")
    
    print("  - Creando tabla temporal con los últimos salarios...")
    cursor.execute("""
    CREATE TEMP TABLE latest_salaries AS
    SELECT DISTINCT ON (emp_no) emp_no, salary
    FROM salaries
    ORDER BY emp_no, to_date DESC;
    """)
    print("  - Tabla temporal creada correctamente.")
    
    print("  - Actualizando salarios en la tabla employees...")
    cursor.execute("""
    UPDATE employees e
    SET salary = ls.salary
    FROM latest_salaries ls
    WHERE e.emp_no = ls.emp_no;
    """)
    print("  - Salarios actualizados correctamente.")
    
    cursor.execute("DROP TABLE latest_salaries;")
    conn.commit()
    print("Columna salary añadida y actualizada en employees")

def get_salary_ranges(conn):
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT 
        MIN(salary) as min_salary,
        MAX(salary) as max_salary,
        percentile_cont(0.25) WITHIN GROUP (ORDER BY salary) as q1,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY salary) as q2,
        percentile_cont(0.75) WITHIN GROUP (ORDER BY salary) as q3
    FROM employees WHERE salary IS NOT NULL;
    """)
    
    min_salary, max_salary, q1, q2, q3 = cursor.fetchone()
    
    return [
        (min_salary, int(q1)),
        (int(q1), int(q2)),
        (int(q2), int(q3)),
        (int(q3), max_salary + 1)
    ]

def create_nested_partitioning(conn):
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS employees3 CASCADE;")
    
    cursor.execute("""
    CREATE TABLE employees3 (
        emp_no INT,
        birth_date DATE,
        first_name VARCHAR(14),
        last_name VARCHAR(16),
        gender CHARACTER(1),
        hire_date DATE,
        dept_no VARCHAR(5),
        from_date DATE,
        salary INTEGER
    ) PARTITION BY RANGE (date_part('year', hire_date));
    """)
    
    cursor.execute("SELECT MIN(date_part('year', hire_date))::int, MAX(date_part('year', hire_date))::int FROM employees;")
    min_year, max_year = cursor.fetchone()

    salary_ranges = get_salary_ranges(conn)
    
    for year in range(min_year, max_year + 1):
        partition_name = f"emp_{year}"
        
        cursor.execute(f"""
        CREATE TABLE {partition_name} PARTITION OF employees3 
        FOR VALUES FROM ({year}) TO ({year+1})
        PARTITION BY RANGE (salary);
        """)
        
        for i, (low_salary, high_salary) in enumerate(salary_ranges):
            cursor.execute(f"""
            CREATE TABLE {partition_name}_s{i} PARTITION OF {partition_name}
            FOR VALUES FROM ({low_salary}) TO ({high_salary});
            """)
    
    conn.commit()
    print("Tabla employees3 creada con fragmentación anidada")

def create_composite_partitioning(conn):
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS employees3_composite CASCADE;")
    
    cursor.execute("""
    CREATE TABLE employees3_composite (
        emp_no INT,
        birth_date DATE,
        first_name VARCHAR(14),
        last_name VARCHAR(16),
        gender CHARACTER(1),
        hire_date DATE,
        dept_no VARCHAR(5),
        from_date DATE,
        salary INTEGER
    ) PARTITION BY RANGE (date_part('year', hire_date), salary);
    """)
    
    cursor.execute("SELECT MIN(date_part('year', hire_date))::int, MAX(date_part('year', hire_date))::int FROM employees;")
    min_year, max_year = cursor.fetchone()
    
    salary_ranges = get_salary_ranges(conn)
    print("Rangos salariales:", salary_ranges)
    
    for year in range(min_year, max_year + 1):
        print(f"  - Creando particiones para el año {year}...")
        
        for i, (low_salary, high_salary) in enumerate(salary_ranges):
            if i < len(salary_ranges) - 1:
                next_low_salary = salary_ranges[i+1][0]
                cursor.execute(f"""
                CREATE TABLE emp_comp_{year}_{i} PARTITION OF employees3_composite
                FOR VALUES FROM ({year}, {low_salary}) TO ({year}, {next_low_salary});
                """)
            else:
                cursor.execute(f"""
                CREATE TABLE emp_comp_{year}_{i} PARTITION OF employees3_composite
                FOR VALUES FROM ({year}, {low_salary}) TO ({year+1}, {salary_ranges[0][0]});
                """)
    
    conn.commit()
    print("Tabla employees3_composite creada con fragmentación compuesta")

def load_partitioned_tables(conn):
    cursor = conn.cursor()
    
    for table_type, table_name in [("anidada", "employees3"), ("compuesta", "employees3_composite")]:
        print(f"  - Cargando tabla con fragmentación {table_type}...")
        cursor.execute(f"""
        INSERT INTO {table_name}
        SELECT emp_no, birth_date, first_name, last_name, gender, hire_date, dept_no, from_date, salary
        FROM employees;
        """)
    
    conn.commit()
    print("Datos cargados en tablas fragmentadas")

def run_query_performance_test(conn, queries):
    cursor = conn.cursor()
    results = {}
    query_results = {
        "query_names": [],
        "regular_times": [],
        "nested_times": [],
        "composite_times": [],
        "best_strategies": []
    }
    
    for query_name, query_variants in queries.items():
        results[query_name] = {}
        
        print(f"\nEjecutando {query_name}: {query_variants['description']}")
        
        for variant_name in ["regular", "nested", "composite"]:
            query_sql = query_variants[variant_name]
            
            cursor.execute(query_sql)
            cursor.fetchall()
            
            start_time = time.time()
            cursor.execute("EXPLAIN ANALYZE " + query_sql)
            plan = cursor.fetchall()
            end_time = time.time()
            
            execution_time = None
            for line in plan:
                if "execution time" in line[0].lower():
                    time_str = line[0].split(":")[1].strip()
                    if "ms" in time_str:
                        execution_time = float(time_str.split(" ")[0])
                    else:
                        execution_time = float(time_str.split(" ")[0]) * 1000
            
            if execution_time is None:
                execution_time = (end_time - start_time) * 1000
            
            table_type = get_table_type(variant_name)
            results[query_name][table_type] = execution_time
            print(f"  {table_type}: {execution_time:.2f} ms")
    
    for query_name, times in results.items():
        regular_time = times["Sin Fragmentación"]
        nested_time = times["Fragmentación Anidada"]
        composite_time = times["Fragmentación Compuesta"]
        
        nested_improvement = ((regular_time - nested_time) / regular_time * 100) if regular_time > 0 else 0
        composite_improvement = ((regular_time - composite_time) / regular_time * 100) if regular_time > 0 else 0
        
        best_strategy = "Anidada" if nested_time <= composite_time else "Compuesta"
        
        query_results["query_names"].append(query_name)
        query_results["regular_times"].append(regular_time)
        query_results["nested_times"].append(nested_time)
        query_results["composite_times"].append(composite_time)
        query_results["best_strategies"].append(best_strategy)
        
        print("{:<10} {:<20.2f} {:<20.2f} ({:.1f}%) {:<20.2f} ({:.1f}%) {:<15}".format(
            query_name, regular_time, nested_time, nested_improvement, 
            composite_time, composite_improvement, best_strategy
        ))
    
    return results, query_results

def save_results_to_csv(results, results_dir, timestamp):
    csv_data = []
    
    for query_name, times in results.items():
        regular_time = times["Sin Fragmentación"]
        nested_time = times["Fragmentación Anidada"]
        composite_time = times["Fragmentación Compuesta"]
        
        nested_improvement = ((regular_time - nested_time) / regular_time * 100) if regular_time > 0 else 0
        composite_improvement = ((regular_time - composite_time) / regular_time * 100) if regular_time > 0 else 0
        
        best_strategy = "Anidada" if nested_time <= composite_time else "Compuesta"
        
        csv_data.append({
            "Consulta": query_name,
            "Sin Fragmentación (ms)": round(regular_time, 2),
            "Fragmentación Anidada (ms)": round(nested_time, 2),
            "Fragmentación Compuesta (ms)": round(composite_time, 2),
            "Mejora Anidada (%)": round(nested_improvement, 1),
            "Mejora Compuesta (%)": round(composite_improvement, 1),
            "Mejor Estrategia": best_strategy
        })
    
    csv_path = os.path.join(results_dir, f"resultados_fragmentacion_{timestamp}.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["Consulta", "Sin Fragmentación (ms)", "Fragmentación Anidada (ms)", 
                     "Fragmentación Compuesta (ms)", "Mejora Anidada (%)", 
                     "Mejora Compuesta (%)", "Mejor Estrategia"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"\nResultados guardados en: {csv_path}")
    return csv_path

def generate_performance_charts(query_results, results_dir, timestamp):
    query_names = query_results["query_names"]
    regular_times = query_results["regular_times"]
    nested_times = query_results["nested_times"]
    composite_times = query_results["composite_times"]
    best_strategies = query_results["best_strategies"]
    
    labels = [f"Q{i+1}" for i in range(len(query_names))]
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    
    bar_width = 0.25
    index = np.arange(len(labels))
    
    bars = [
        plt.bar(index - bar_width, regular_times, bar_width, label='Sin Fragmentación', color='blue', alpha=0.7),
        plt.bar(index, nested_times, bar_width, label='Fragmentación Anidada', color='green', alpha=0.7),
        plt.bar(index + bar_width, composite_times, bar_width, label='Fragmentación Compuesta', color='orange', alpha=0.7)
    ]
    
    for i, v in enumerate(regular_times):
        plt.text(i - bar_width, v + 5, f"{v:.1f}", ha='center', va='bottom', fontsize=8, rotation=0)
    
    for i, v in enumerate(nested_times):
        plt.text(i, v + 5, f"{v:.1f}", ha='center', va='bottom', fontsize=8, rotation=0)
        
    for i, v in enumerate(composite_times):
        plt.text(i + bar_width, v + 5, f"{v:.1f}", ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.xlabel('Consultas')
    plt.ylabel('Tiempo de ejecución (ms)')
    plt.title('Comparación de tiempos de ejecución por tipo de fragmentación')
    plt.xticks(index, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    
    nested_improvements = [(regular - nested) / regular * 100 if regular > 0 else 0 
                          for regular, nested in zip(regular_times, nested_times)]
    
    composite_improvements = [(regular - composite) / regular * 100 if regular > 0 else 0 
                             for regular, composite in zip(regular_times, composite_times)]
    
    plt.bar(index - bar_width/2, nested_improvements, bar_width, label='Mejora Anidada (%)', color='green', alpha=0.7)
    plt.bar(index + bar_width/2, composite_improvements, bar_width, label='Mejora Compuesta (%)', color='orange', alpha=0.7)
    
    for i, v in enumerate(nested_improvements):
        plt.text(i - bar_width/2, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
        
    for i, v in enumerate(composite_improvements):
        plt.text(i + bar_width/2, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
    
    for i, strategy in enumerate(best_strategies):
        color = 'green' if strategy == 'Anidada' else 'orange'
        plt.annotate(f"✓ {strategy}", 
                    xy=(i, max(nested_improvements[i], composite_improvements[i]) + 10),
                    ha='center', va='bottom', color=color, weight='bold')
    
    plt.xlabel('Consultas')
    plt.ylabel('Mejora porcentual (%)')
    plt.title('Porcentaje de mejora respecto a la consulta sin fragmentación')
    plt.xticks(index, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    chart_path = os.path.join(results_dir, f"grafico_rendimiento_{timestamp}.png")
    plt.savefig(chart_path)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(index, regular_times, 'o-', label='Sin Fragmentación')
    plt.semilogy(index, nested_times, 's-', label='Fragmentación Anidada')
    plt.semilogy(index, composite_times, '^-', label='Fragmentación Compuesta')
    
    for i, (reg, nest, comp) in enumerate(zip(regular_times, nested_times, composite_times)):
        plt.annotate(f"{reg:.1f}", (i, reg), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f"{nest:.1f}", (i, nest), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f"{comp:.1f}", (i, comp), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Consultas')
    plt.ylabel('Tiempo de ejecución (ms) - Escala logarítmica')
    plt.title('Comparación de rendimiento (escala logarítmica)')
    plt.xticks(index, labels)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    log_chart_path = os.path.join(results_dir, f"grafico_rendimiento_log_{timestamp}.png")
    plt.savefig(log_chart_path)
    plt.close()
    
    return chart_path, log_chart_path

def save_execution_plans(conn, queries, results_dir, timestamp):
    cursor = conn.cursor()
    plans_data = []
    
    for query_name, query_variants in queries.items():
        print(f"\nPlanes de ejecución para {query_name}")
        
        for variant_name in ["regular", "nested", "composite"]:
            table_type = get_table_type(variant_name)
            query_sql = query_variants[variant_name]
            
            cursor.execute("EXPLAIN ANALYZE " + query_sql)
            plan = cursor.fetchall()
            plan_text = "\n".join([line[0] for line in plan])
            
            execution_time = None
            for line in plan:
                if "execution time" in line[0].lower():
                    time_str = line[0].split(":")[1].strip()
                    if "ms" in time_str:
                        execution_time = float(time_str.split(" ")[0])
                    else:
                        execution_time = float(time_str.split(" ")[0]) * 1000
            
            plans_data.append({
                "Consulta": query_name,
                "Tipo": table_type,
                "Tiempo_ms": execution_time,
                "Plan": plan_text
            })
    
    plans_csv_path = os.path.join(results_dir, f"planes_ejecucion_{timestamp}.csv")
    with open(plans_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Consulta", "Tipo", "Tiempo_ms", "Plan"])
        writer.writeheader()
        for row in plans_data:
            writer.writerow(row)
    
    summary_csv_path = os.path.join(results_dir, f"resumen_planes_{timestamp}.csv")
    with open(summary_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["Consulta", "Sin Fragmentación (ms)", "Fragmentación Anidada (ms)", "Fragmentación Compuesta (ms)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for query_name in set(row["Consulta"] for row in plans_data):
            row_data = {"Consulta": query_name}
            
            for row in plans_data:
                if row["Consulta"] == query_name:
                    if row["Tipo"] == "Sin Fragmentación":
                        row_data["Sin Fragmentación (ms)"] = row["Tiempo_ms"]
                    elif row["Tipo"] == "Fragmentación Anidada":
                        row_data["Fragmentación Anidada (ms)"] = row["Tiempo_ms"]
                    elif row["Tipo"] == "Fragmentación Compuesta":
                        row_data["Fragmentación Compuesta (ms)"] = row["Tiempo_ms"]
            
            writer.writerow(row_data)
    
    print(f"Planes de ejecución guardados en: {plans_csv_path}")
    print(f"Resumen de planes guardado en: {summary_csv_path}")
    
    return plans_csv_path, summary_csv_path

def main():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensure_dir(RESULTS_DIR)
        
        conn = connect_to_postgres()
        
        print("Iniciando carga de datos desde archivos CSV...")
        extract_and_load_data(conn)
        print("Datos base cargados exitosamente.")
        
        print("Añadiendo columna de salario a los empleados...")
        add_salary_column(conn)
        print("Columna de salario añadida correctamente.")
        
        print("Creando tabla con fragmentación anidada (puede tardar unos minutos)...")
        create_nested_partitioning(conn)
        print("Fragmentación anidada completada.")
        
        print("Creando tabla con fragmentación compuesta (puede tardar unos minutos)...")
        create_composite_partitioning(conn)
        print("Fragmentación compuesta completada.")
        
        print("Cargando datos en las tablas fragmentadas...")
        load_partitioned_tables(conn)
        print("Datos cargados en tablas fragmentadas exitosamente.")
        
        print("Iniciando pruebas de rendimiento...")
        print("\n" + "="*80)
        print("RESULTADOS COMPARATIVOS")
        print("="*80)
        print("{:<10} {:<20} {:<20} {:<20} {:<15}".format(
            "Consulta", "Sin Fragmentación (ms)", "Fragm. Anidada (ms)", "Fragm. Compuesta (ms)", "Mejor Estrategia"
        ))
        print("-"*80)
        
        results, query_results = run_query_performance_test(conn, QUERIES)
        
        print("-"*80)
        print("Nota: Los valores entre paréntesis indican el porcentaje de mejora respecto a la consulta sin fragmentación")
        
        save_results_to_csv(results, RESULTS_DIR, timestamp)
        generate_performance_charts(query_results, RESULTS_DIR, timestamp)
        
        print("\nCapturando planes de ejecución detallados...")
        save_execution_plans(conn, QUERIES, RESULTS_DIR, timestamp)
        print("Planes de ejecución guardados correctamente.")
        
        print("Cerrando conexión a la base de datos...")
        conn.close()
        print("Proceso completado con éxito.")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        print("El proceso se ha detenido debido a un error.")

main()