Biblioteca utilizada localmente:
```py
import os
import sys

project_path = os.path.abspath(os.path.join(os.getcwd(), "..")) 
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import files
```