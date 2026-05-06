# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for building standalone RamaLama executable on macOS.

This creates a single executable that includes Python and all dependencies,
eliminating the need for users to install Python separately.

Install the package first so entry-point metadata is available (copy_metadata):
  pip install .

Build with: pyinstaller ramalama.spec
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# Import version dynamically
sys.path.insert(0, str(Path.cwd() / 'ramalama'))
from version import version as get_version

# Get the project root directory
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

# Every submodule under ramalama (plugins, transports, etc.) for PyInstaller analysis.
_ramalama_hiddenimports = collect_submodules('ramalama')

# Get version dynamically
app_version = get_version()

# Collect all ramalama package files
a = Analysis(
    ['bin/ramalama'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # Include shortnames configuration
        ('shortnames/shortnames.conf', 'share/ramalama'),
        # Include ramalama.conf
        ('docs/ramalama.conf', 'share/ramalama'),
        # Include completions
        ('completions/bash-completion/completions/*', 'share/bash-completion/completions'),
        ('completions/fish/vendor_completions.d/*', 'share/fish/vendor_completions.d'),
        ('completions/zsh/site-functions/*', 'share/zsh/site-functions'),
        # Include man pages
        ('docs/*.1', 'share/man/man1'),
        ('docs/*.5', 'share/man/man5'),
        ('docs/*.7', 'share/man/man7'),
    ]
    + copy_metadata('ramalama'),  # dist metadata for importlib.metadata.entry_points (runtime plugins)
    hiddenimports=_ramalama_hiddenimports
    + [
        'argcomplete',
        'yaml',
        'jinja2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'PIL',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ramalama',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disabled for Apple Silicon compatibility
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ramalama',
)

app = BUNDLE(
    coll,
    name='ramalama.app',
    icon='logos/ICNS/ramalama.icns',
    bundle_identifier='com.github.containers.ramalama',
    version=app_version,
    info_plist={
        'CFBundleName': 'RamaLama',
        'CFBundleDisplayName': 'RamaLama',
        'CFBundleIdentifier': 'com.github.containers.ramalama',
        'CFBundleVersion': app_version,
        'CFBundleShortVersionString': app_version,
        'NSHumanReadableCopyright': 'Copyright © 2026 The Containers Organization',
        'CFBundleExecutable': 'ramalama',
    },
)
