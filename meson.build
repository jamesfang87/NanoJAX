project(
  'NanoJAX',
  'cpp',
  version: '0.0.1',
  meson_version:'>=1.0.0',
  default_options: ['cpp_std=c++23', 'buildtype=debug', 'b_lto=true']
)

nanojax_incdir = include_directories('include')

nanojax_lib = static_library(
  'nanojax',
  ['src/ad/scalar.cpp', 'src/nn/data/dataset.cpp'],
  include_directories: nanojax_incdir,
)

nanojax_exe = executable(
  'nanojax',
  ['src/main.cpp'],
  include_directories: nanojax_incdir,
  link_with: nanojax_lib,
)

foreach name : [
  'test_scalar', 'test_variable', 'test_grad', 'test_gradcheck', 'test_loss',
  'test_vecmat', 'test_mlp'
]
  test(
    name,
    executable(
      name,
      ['tests/@0@.cpp'.format(name)],
      include_directories: nanojax_incdir,
      link_with: nanojax_lib,
    ),
  )
endforeach
