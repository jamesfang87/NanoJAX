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

py.install_sources(
  'nanojax/nn/__init__.py',
  'nanojax/nn/activations.py',
  'nanojax/nn/attn.py',
  'nanojax/nn/cnn.py',
  'nanojax/nn/flatten.py',
  'nanojax/nn/layernorm.py',
  'nanojax/nn/loss.py',
  'nanojax/nn/mlp.py',
  'nanojax/nn/module.py',
  subdir: 'nanojax/nn',
)

py.install_sources(
  'nanojax/nn/optimizers/__init__.py',
  'nanojax/nn/optimizers/optim.py',
  'nanojax/nn/optimizers/sgd.py',
  subdir: 'nanojax/nn/optimizers',
)

py.install_sources(
  'nanojax/nn/data/__init__.py',
  'nanojax/nn/data/dataset.py',
  'nanojax/nn/data/dataloader.py',
  subdir: 'nanojax/nn/data',
)

py.install_sources(
  'nanojax/nn/data/examples/__init__.py',
  'nanojax/nn/data/examples/mnist.py',
  subdir: 'nanojax/nn/data/examples',
)
