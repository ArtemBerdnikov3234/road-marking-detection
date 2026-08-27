import os
import shutil
import yaml
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Права доступа (разрешаем загрузку и управление файлами, созданными этим скриптом)
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_drive_service():
    """Аутентификация и создание сервиса Google Drive."""
    creds = None
    # Файл token.json хранит токены доступа и обновления пользователя
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # Если нет валидных токенов, просим пользователя войти
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                raise FileNotFoundError(
                    "Файл credentials.json не найден.\n"
                    "Пожалуйста, скачайте его из Google Cloud Console (APIs & Services -> Credentials)\n"
                    "и положите в папку со скриптом."
                )
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Сохраняем токен для будущих запусков
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

def find_or_create_folder(service, folder_name, parent_id=None):
    """Ищет папку по имени, если нет - создает."""
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
        
    results = service.files().list(q=query, spaces='drive', fields='nextPageToken, files(id, name)').execute()
    items = results.get('files', [])

    if not items:
        print(f"Папка '{folder_name}' не найдена. Создаю...")
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            folder_metadata['parents'] = [parent_id]
            
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        return folder.get('id')
    else:
        print(f"Найдена папка '{folder_name}' (ID: {items[0]['id']})")
        return items[0]['id']

def upload_file(service, filepath, folder_id):
    """Загружает файл в указанную папку Google Drive."""
    filename = os.path.basename(filepath)
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    media = MediaFileUpload(filepath, resumable=True)
    
    print(f"Начинаю загрузку {filename} в Google Drive...")
    request = service.files().create(body=file_metadata, media_body=media, fields='id')
    
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Прогресс загрузки: {int(status.progress() * 100)}%")
            
    print(f"Файл успешно загружен. File ID: {response.get('id')}")

def main():
    # 1. Загрузка конфига
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / "config.yaml"
    
    if not config_path.exists():
        print(f"Ошибка: Не найден {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Определяем пути к данным (в данном случае берём готовый yolo dataset из конфига)
    root_dir = (script_dir / config['project']['root_dir']).resolve()
    dataset_dir = root_dir / config['project']['output_dataset_dir']
    
    if not dataset_dir.exists():
        print(f"Ошибка: Директория с датасетом не найдена: {dataset_dir}")
        return
        
    zip_filename = root_dir / f"{dataset_dir.name}.zip"
    
    # 3. Архивируем папку
    print(f"Архивируем {dataset_dir} -> {zip_filename}")
    shutil.make_archive(
        base_name=str(zip_filename.with_suffix('')),
        format='zip',
        root_dir=dataset_dir
    )
    print("Архивация завершена.")

    # 4. Аутентификация в Google Drive
    print("Подключаемся к Google Drive API...")
    try:
        service = get_drive_service()
    except Exception as e:
        print(f"Ошибка аутентификации: {e}")
        return

    # 5. Поиск или создание папки SibDor
    folder_name = "SibDor"
    folder_id = find_or_create_folder(service, folder_name)

    # 6. Загрузка файла
    upload_file(service, str(zip_filename), folder_id)
    
    print("Всё готово!")

if __name__ == '__main__':
    main()
