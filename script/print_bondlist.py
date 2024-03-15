
atom_list =  ['O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']

print("&atomlist")
for i,j in enumerate(atom_list):
    print(i, j)


bonds_list = [[0, 5], [1, 0], [1, 6], [1, 2], [2, 9], [3, 2], [3, 10], [3, 4], [4, 12], [7, 1], [8, 2], [11, 3], [13, 4], [14, 4]]

print("&bondlist")
for i in bonds_list:
    print(i[0], i[1])


print("&representative")
print(2)
