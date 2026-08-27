import argparse
import shutil
import sys
from pathlib import Path
import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Запаковка готового датасета в ZIP-архив.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Путь к файлу конфигурации")
    args = parser.parse_args()

    # Загружаем конфиг
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"[ОШИБКА] Конфиг не найден: {config_path}")
        sys.exit(1)
        
    cfg = load_config(str(config_path))
    
    # Определяем пути
    root_dir = Path(__file__).parent / cfg["project"]["root_dir"]
    dataset_dir = (root_dir / cfg["project"]["output_dataset_dir"]).resolve()
    
    # Мы пакуем именно отфильтрованный 'horizontal' датасет
    target_dir = dataset_dir / "horizontal"
    
    if not target_dir.exists():
        print(f"[ОШИБКА] Папка с датасетом не найдена: {target_dir}")
        sys.exit(1)
        
    # Имя архива будет horizontal_dataset.zip, сохраняем его в корень проекта (root_dir)
    # или можно рядом с папкой 02_yolo_dataset
    zip_path_base = root_dir / "horizontal_dataset"
    
    print(f"{'='*65}")
    print(f"  Упаковка датасета в ZIP-архив")
    print(f"{'='*65}")
    print(f"  Исходная папка: {target_dir}")
    print(f"  Создание архива... Это может занять пару минут.")
    
    try:
        # make_archive добавляет расширение .zip автоматически
        shutil.make_archive(
            base_name=str(zip_path_base),
            format='zip',
            root_dir=str(target_dir)
        )
        final_zip = zip_path_base.with_suffix(".zip")
        print(f"  [OK] Архив успешно создан: {final_zip}")
        print(f"  Размер архива: {final_zip.stat().st_size / (1024*1024):.1f} МБ")
    except Exception as e:
        print(f"  [ОШИБКА] Не удалось создать архив: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
