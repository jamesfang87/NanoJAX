project(
  'NanoJAX',
  'cpp',
  version: '0.0.1',
  meson_version:'>=1.0.0',
  default_options: ['cpp_std=c++23', 'buildtype=debug']
)

nanojax_exe = executable(
  'nanojax',
  ['src/main.cpp', 'src/ad/scalar.cpp'],
  include_directories: include_directories('.'),
)
