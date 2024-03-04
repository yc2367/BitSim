def int_to_2s_complement(value: int, bitwidth: int):
  bin_list = []
  if value >= 0:
    for i in reversed(range(bitwidth)):
      if value >= 2**i:
        bin_list.append('1')
        value -= 2**i
      else:
        bin_list.append('0')
  else:
    bin_list.append('1')
    value += 2**(bitwidth - 1)
    for i in reversed(range(bitwidth - 1)):
      if value >= 2**i:
        bin_list.append('1')
        value -= 2**i
      else:
        bin_list.append('0')

  return '  '.join(bin_list)


def int_to_sign_magnitude(value: int, bitwidth: int):
  if value == -128:
    value = -127
  bin_list = []
  if value >= 0:
    bin_list.append('0')
    for i in reversed(range(bitwidth - 1)):
      if value >= 2**i:
        bin_list.append('1')
        value -= 2**i
      else:
        bin_list.append('0')
  else:
    bin_list.append('1')
    value = abs(value)
    for i in reversed(range(bitwidth - 1)):
      if value >= 2**i:
        bin_list.append('1')
        value -= 2**i
      else:
        bin_list.append('0')

  return '  '.join(bin_list)

