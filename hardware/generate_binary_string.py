bs_list = []

def gen_binary(bit, bs=''):
	if bit > 0:
		gen_binary(bit-1, bs+'0')
		gen_binary(bit-1, bs+'1')
	else:
		bs_list.append(bs)

gen_binary(5)
print(bs_list)

for i in range(17):
	print("5\'b" + bs_list[i] + ": out = vec[" + str(i) + "];")