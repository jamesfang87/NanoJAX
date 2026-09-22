py = import('python').find_installation(
  meson.project_source_root() + '/.venv/bin/python',
  pure: false,
)

nanobind_include = run_command(
  py, ['-c', 'import nanobind; print(nanobind.include_dir())'], check: true,
).stdout().strip()

nanobind_source = run_command(
  py, ['-c', 'import nanobind; print(nanobind.source_dir())'], check: true,
).stdout().strip()

nanobind_runtime = custom_target(
  'nanobind_runtime',
  output: 'nb_combined.cpp',
  command: ['cp', nanobind_source + '/nb_combined.cpp', '@OUTPUT@'],
)

nanobind_args = [
  '-I' + nanobind_include,
  '-I' + nanobind_source,
  '-I' + nanobind_source + '/../ext/robin_map/include',
  '-fvisibility=hidden',
  '-fno-strict-aliasing',
]

py.extension_module(
  '_tensor',
  'module.cpp',
  nanobind_runtime,
  include_directories: inc,
  dependencies: nanojax_deps,
  cpp_args: nanobind_args,
  subdir: 'nanojax',
  install: true,
)
