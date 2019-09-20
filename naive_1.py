import datareader


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
    for i in range(2**16):
        for j in range(2**16):
            if mat1.get(i, None) is not None and mat1[i].get(j, None):
                res.setdefault(i, {})
                res[i][j] = True
            elif mat2.get(i, None) is not None and mat2[i].get(j, None):
                res.setdefault(i, {})
                res[i][j] = True



ids_attack_free = [message.id for message in datareader_old.load_attack_free1(0)]
ids_attack_free_2 = [message.id for message in datareader_old.load_attack_free2(0)]
ids_fuzzy_bad = [message.id for message in datareader_old.load_fuzzy(500000, 50)]
ids_dos = [message.id for message in datareader_old.load_dos(0, 50)]
ids_imp = [message.id for message in datareader_old.load_impersonation_1(0, 50)]

mat1 = build_transition_matrix(ids_attack_free)
mat2 = build_transition_matrix(ids_attack_free_2)
matrix = combine_matrix(mat1, mat2)
print(is_intrusion(ids_attack_free, matrix))
print(is_intrusion(ids_attack_free_2, matrix))
print(is_intrusion(ids_fuzzy_bad, matrix))
print(is_intrusion(ids_dos, matrix))
print(is_intrusion(ids_imp, matrix))

ids_fuzzy_good_1 = [message.id for message in datareader_old.load_fuzzy(0, 100000)]
ids_fuzzy_good_2 = [message.id for message in datareader_old.load_fuzzy(100000, 1000)]

matrix_fuzzy = build_transition_matrix(ids_fuzzy_good_1)
print(is_intrusion(ids_fuzzy_good_2, matrix_fuzzy))