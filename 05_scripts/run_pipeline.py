import subprocess
import sys
import argparse
from pathlib import Path
import yaml

def update_config_raw_data(config_path, new_raw_data_path):
    print(f"Обновление config.yaml: устанавливаем raw_data_dir = {new_raw_data_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Можно передавать как абсолютный путь, так и относительный.
    # Если путь абсолютный, сохраняем как есть.
    # Если внутри проекта - можно сделать относительным, но конфиг поддерживает и абсолютные,
    # так как во всех скриптах путь скорее всего склеивается как root_dir / raw_data_dir.
    # В идеале конвертировать в путь относительно корня проекта, если это возможно,
    # либо просто записать как строку. Чтобы не ломать логику `root_dir / raw_data_dir` 
    # в других скриптах, если передали абсолютный путь, лучше попытаться сделать его относительным
    # к root_dir (который по умолчанию "..").
    
    # Для простоты и надежности: запишем переданный путь как абсолютный, либо как есть.
    # В python pathlib: Path(root_dir) / Path(абсолютный_путь) = абсолютный_путь.
    # Так что абсолютные пути будут работать корректно.
    
    config['project']['raw_data_dir'] = new_raw_data_path
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Запуск: {script_name}")
    print(f"{'='*50}")
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"[ОШИБКА] Скрипт {script_name} не найден!")
        sys.exit(1)
    result = subprocess.run([sys.executable, str(script_path)], cwd=script_path.parent)
    
    if result.returncode != 0:
        print(f"\n[ОШИБКА] Скрипт {script_name} завершился с ошибкой (код {result.returncode}).")
        print("Остановка пайплайна.")
        sys.exit(result.returncode)
        
    print(f"\n[OK] Скрипт {script_name} успешно выполнен.")

def main():
    parser = argparse.ArgumentParser(description="Запуск полного пайплайна обработки датасета.")
    parser.add_argument("--raw_data", type=str, help="Путь к исходному датасету (например: E:\\project\\SibDor\\01_raw_data\\SibDor_markings_dataset)")
    args = parser.parse_args()

    if args.raw_data:
        config_file = Path(__file__).parent / "config.yaml"
        update_config_raw_data(config_file, args.raw_data)

    scripts_to_run = [
        "01_eda_raw.py",
        "02_split_dataset.py",
        "03_validate_dataset.py",
        "04_filter_horizontal_only.py",
        "05_oversample_rare_classes.py",
        "06_zip_dataset.py"
    ]
    
    for script in scripts_to_run:
        run_script(script)
        
    print("\n[OK] Весь пайплайн (01-06) успешно завершен!")

if __name__ == "__main__":
    main()
