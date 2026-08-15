project(
  'NanoJAX',
  'cpp',
  version: '0.0.1',
  meson_version:'>=1.0.0',
  default_options: ['cpp_std=c++23', 'buildtype=debug']
)

nanojax_incdir = include_directories('include')

nanojax_lib = shared_library(
  'nanojax',
  ['src/ad/scalar.cpp'],
  include_directories: nanojax_incdir,
)

nanojax_exe = executable(
  'nanojax',
  ['src/main.cpp'],
  include_directories: nanojax_incdir,
  link_with: nanojax_lib,
)

# Every test executable links against the shared library rather than
# recompiling src/ad/scalar.cpp directly.
foreach name : ['test_scalar', 'test_variable', 'test_grad']
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
