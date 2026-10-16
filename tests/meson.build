tensor_test = executable(
  'tensor_test',
  'tensor_test.cpp',
  include_directories: inc,
  dependencies: nanojax_deps,
)

test('tensor', tensor_test)

matmul_grad_test = executable(
  'matmul_grad_test',
  'matmul_grad_test.cpp',
  include_directories: inc,
  dependencies: nanojax_deps,
)

test('matmul_grad', matmul_grad_test)

copy_grad_test = executable(
  'copy_grad_test',
  'copy_grad_test.cpp',
  include_directories: inc,
  dependencies: nanojax_deps,
)

test('copy_grad', copy_grad_test)

view_grad_test = executable(
  'view_grad_test',
  'view_grad_test.cpp',
  include_directories: inc,
  dependencies: nanojax_deps,
)

test('view_grad', view_grad_test)

conv_test = executable(
  'conv_test',
  'conv_test.cpp',
  include_directories: inc,
  dependencies: nanojax_deps,
)

test('conv', conv_test)

slice_grad_test = executable(
  'slice_grad_test',
  'slice_grad_test.cpp',
  include_directories: inc,
  dependencies: nanojax_deps,
)

test('slice_grad', slice_grad_test)
