import datareader_csv


def build_transition_matrix(ids):
    matrix = {}

    for i in range(1, len(ids)):
        if matrix.get(ids[i], None) is None:
            matrix[ids[i]] = {}
        
        matrix[ids[i]][ids[i-1]] = True
        
    return matrix


def is_intrusion(ids, matrix):
    for i in range(1, len(ids)):
        if matrix.get(ids[i], None) is None or matrix[ids[i]].get(ids[i - 1], None) is None:
            return True
        
    return False


def combine_matrix(mat1, mat2):
    res = {}
    for row in mat1.keys():
        for col in mat1[row].keys():
            res.setdefault(row, {})
            res[row][col] = mat1[row][col]

    for row in mat2.keys():
        for col in mat2[row].keys():
            res.setdefault(row, {})
            res[row][col] = mat2[row][col]

    return res


ids_attack_free = [message.id for message in datareader_csv.load_attack_free1(0, 10000)]
ids_attack_free_2 = [message.id for message in datareader_csv.load_attack_free2(0, 10000)]
ids_fuzzy_bad = [message.id for message in datareader_csv.load_fuzzy(500000, 50)]
ids_dos = [message.id for message in datareader_csv.load_dos(0, 50)]
ids_imp = [message.id for message in datareader_csv.load_impersonation_1(0, 50)]

mat1 = build_transition_matrix(ids_attack_free)
mat2 = build_transition_matrix(ids_attack_free_2)
matrix = combine_matrix(mat1, mat2)
print(is_intrusion(ids_attack_free, matrix))
print(is_intrusion(ids_attack_free_2, matrix))
print(is_intrusion(ids_fuzzy_bad, matrix))
print(is_intrusion(ids_dos, matrix))
print(is_intrusion(ids_imp, matrix))

ids_fuzzy_good_1 = [message.id for message in datareader_csv.load_fuzzy(0, 100000)]
ids_fuzzy_good_2 = [message.id for message in datareader_csv.load_fuzzy(100000, 1000)]

matrix_fuzzy = build_transition_matrix(ids_fuzzy_good_1)
print(is_intrusion(ids_fuzzy_good_2, matrix_fuzzy))