import random

def calculate_sparsity(entity_count, relation_count, triples_count):
   
    sparsity = triples_count / (entity_count * (entity_count - 1) * relation_count)
    return sparsity

def sparse_relations(input_file, output_file, sparsity_rate):
   
    relation_counts = {}

 
    entity_counts = {}

    
    # with open(input_file, 'r') as infile:
    #     for line in infile:
    #         triple = line.strip().split('\t')
    #         if len(triple) == 3:
    #             head, relation, tail = triple
    #             relation_counts[relation] = relation_counts.get(relation, 0) + 1
    #             entity_counts[head] = entity_counts.get(head, 0) + 1
    #             entity_counts[tail] = entity_counts.get(tail, 0) + 1

    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            triple = line.strip().split('\t')
            if len(triple) == 3:
                head, relation, tail = triple
                if (head not in entity_counts) | (tail not in entity_counts):
                    outfile.write('\t'.join([head, relation, tail]) + '\n')

                    entity_counts[head] = entity_counts.get(head, 0) + 1
                    entity_counts[tail] = entity_counts.get(tail, 0) + 1
                elif random.random() > sparsity_rate:
                    outfile.write('\t'.join([head, relation, tail]) + '\n')

                    entity_counts[head] = entity_counts.get(head, 0) + 1
                    entity_counts[tail] = entity_counts.get(tail, 0) + 1
               
                # relation_density = calculate_sparsity(len(entity_counts), relation_counts[relation], len(relation_counts))

def change_relation_order(input_file_path, output_file_path):
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            triple = line.strip().split('\t')
            if len(triple) == 3:
                head, tail, relation = triple
                outfile.write('\t'.join([head, relation, tail]) + '\n')

# 
# input_file_path = '../data/FB15k-237/train.txt'
# output_file_path = '../data/FB15k-237/train100p.txt'
# sparsity_rate = 0.4  
#
# 
# sparse_relations(input_file_path, output_file_path, sparsity_rate)
input_file_path = '../data/FB15K/test.txt'
output_file_path = '../data/FB15K/test100p.txt'
change_relation_order(input_file_path, output_file_path)
