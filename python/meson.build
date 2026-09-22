py = import('python').find_installation(
  meson.project_source_root() + '/.venv/bin/python',
  pure: false,
)

py.install_sources(
  'nanojax/__init__.py',
  'nanojax/_tensor.pyi',
  'nanojax/transform.py',
  subdir: 'nanojax',
)
