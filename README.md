# Dimenfix TSNE

### Run Commands

**Make sure to run in cmd, not powershell.**

pip freeze > requirements.txt


enable virtual env
```cmd
python -m venv myenv
myenv\Scripts\activate
```

Change to install path of VS

```cmd
SET DISTUTILS_USE_SDK=1
"D:\VS\VC\Auxiliary\Build\vcvarsall.bat" x64

"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
```

Install dependencies
```cmd
pip install -r requirements.txt
pip install numpy meson meson-python scipy cython
```

Install editable build
```cmd
pip install --editable . --verbose --no-build-isolation --config-settings editable-verbose=true
```

**README.rst** is required for installation.