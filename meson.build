project(
  'nanojax',
  'cpp',
  version: '0.1.0',
  meson_version: '>=1.5.0',
  default_options: [
    'cpp_std=c++23',
    'cuda_std=c++23',
    'warning_level=2',
    'buildtype=release',
  ],
)

add_project_arguments(
  '-fno-math-errno',
  '-ffast-math',
  '-funroll-loops',
  '-march=native',
  language: 'cpp',
)

have_cuda = add_languages('cuda', required: false, native: false)

inc = include_directories('include')

boost_dep = dependency(
  'boost',
  modules: ['container'],
)

blas_dep = dependency('appleframeworks', modules: ['Accelerate'], required: false)
blas_is_accelerate = blas_dep.found()

if not blas_dep.found()
  foreach name : ['openblas', 'cblas']
    blas_dep = dependency(name, required: false)
    if blas_dep.found()
      break
    endif
  endforeach
endif

nanojax_deps = [boost_dep]
if blas_dep.found()
  add_project_arguments('-DNANOJAX_HAVE_CBLAS', language: 'cpp')
  if blas_is_accelerate
    add_project_arguments('-DACCELERATE_NEW_LAPACK', language: 'cpp')
  endif
  nanojax_deps += blas_dep
endif

nanojax_dep = declare_dependency(
  include_directories: inc,
  dependencies: nanojax_deps,
)

subdir('tests')
subdir('bindings')
subdir('python')
