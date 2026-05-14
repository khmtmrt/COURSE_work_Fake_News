import os
from pathlib import Path

def manual_copy():
    source_dir = Path(r"c:\навчання\третій курс\2 семестр 3 курс\Машинне навчання\КУРСОВА\ПОГАНА КУРСОВА")
    target_dir = Path(r"c:\навчання\третій курс\2 семестр 3 курс\Машинне навчання\КУРСОВА\ДОБРА КУРСОВА")

    target_dir.mkdir(parents=True, exist_ok=True)
    
    files_copied = 0
    folders_created = 0
    
    for root, dirs, files in os.walk(source_dir):
        current_source_path = Path(root)
        relative_path = current_source_path.relative_to(source_dir)
        current_target_path = target_dir / relative_path
        
        for d in dirs:
            dir_target_path = current_target_path / d
            if not dir_target_path.exists():
                dir_target_path.mkdir(parents=True, exist_ok=True)
                folders_created += 1
                
        for file in files:
            source_file = current_source_path / file
            target_file = current_target_path / file
            
            with open(source_file, 'rb') as f_src:
                content = f_src.read()
                
            with open(target_file, 'wb') as f_dst:
                f_dst.write(content)
                
            files_copied += 1
            
    print(f"Успішно переписано файлів: {files_copied}")
    print(f"Успішно створено папок: {folders_created}")

if __name__ == "__main__":
    manual_copy()
