import os

def print_folder_structure(start_path='.', indent=''):
    for item in sorted(os.listdir(start_path)):
        path = os.path.join(start_path, item)
        print(indent + '|-- ' + item)
        if os.path.isdir(path):
            print_folder_structure(path, indent + '    ')

print("Folder structure from current directory:\n")
print_folder_structure('.')