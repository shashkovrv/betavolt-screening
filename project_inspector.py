import os
from pathlib import Path

# Игнорируем системный мусор
IGNORE_DIRS = {".venv", "venv", "__pycache__", ".git", ".idea", ".vscode"}

def scan_project(root_path="."):
    root = Path(root_path).resolve()
    print(f"\n=======================================================")
    print(f" СНИМОК ПРОЕКТА: {root.name}")
    print(f" Путь: {root}")
    print(f"=======================================================\n")
    
    empty_files = []
    filled_files = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Отсекаем .venv и кэши
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        
        rel_dir = Path(dirpath).relative_to(root)
        depth = len(rel_dir.parts)
        indent = "  " * depth
        
        if rel_dir != Path("."):
            print(f"{indent}📁 {rel_dir.name}/")
        
        sub_indent = "  " * (depth + 1)
        for f in sorted(filenames):
            if f == "inspect_project.py":
                continue
            file_path = Path(dirpath) / f
            size = file_path.stat().st_size
            rel_file = rel_dir / f

            if size == 0:
                status = "🔴 [ПУСТОЙ]"
                empty_files.append(str(rel_file))
            elif size < 1024:
                status = f"🟢 [{size} B]"
                filled_files.append(str(rel_file))
            elif size < 1024 * 1024:
                status = f"🟢 [{size / 1024:.1f} KB]"
                filled_files.append(str(rel_file))
            else:
                status = f"💾 [{size / (1024*1024):.1f} MB]"
                filled_files.append(str(rel_file))
                
            print(f"{sub_indent}{f:<35} {status}")

    print(f"\n=======================================================")
    print(f"ИТОГО: Файлов с кодом/данными: {len(filled_files)} | Пустых файлов: {len(empty_files)}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    scan_project()