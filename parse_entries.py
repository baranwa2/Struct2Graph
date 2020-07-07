# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:25:40 2020
@author: mayank
"""

from Bio import SeqIO
import numpy as np
import os
import pickle
from os import walk
from numpy import linalg as LA
from collections import defaultdict

all_amino = []

max_residues = 2000

amino_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','PYL','SER','SEC','THR','TRP','TYR','VAL','ASX','GLX','XAA','XLE']

amino_short = {}
amino_short['ALA'] = 'A'
amino_short['ARG'] = 'R'
amino_short['ASN'] = 'N'
amino_short['ASP'] = 'D'
amino_short['CYS'] = 'C'
amino_short['GLN'] = 'Q'
amino_short['GLU'] = 'E'
amino_short['GLY'] = 'G'
amino_short['HIS'] = 'H'
amino_short['ILE'] = 'I'
amino_short['LEU'] = 'L'
amino_short['LYS'] = 'K'
amino_short['MET'] = 'M'
amino_short['PHE'] = 'F'
amino_short['PRO'] = 'P'
amino_short['PYL'] = 'O'
amino_short['SER'] = 'S'
amino_short['SEC'] = 'U'
amino_short['THR'] = 'T'
amino_short['TRP'] = 'W'
amino_short['TYR'] = 'Y'
amino_short['VAL'] = 'V'
amino_short['ASX'] = 'B'
amino_short['GLX'] = 'Z'
amino_short['XAA'] = 'X'
amino_short['XLE'] = 'J'





def create_fingerprints(atoms, adjacency, radius):
    """Extract r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using WeisfeilerLehman-like algorithm."""
    
    fingerprints = []
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        for i in range(len(atoms)):
            vertex      = atoms[i]
            neighbors   = tuple(set(tuple(sorted(atoms[np.where(adjacency[i]>0.0001)[0]]))))
            fingerprint = (vertex, neighbors)
            fingerprints.append(fingerprint_dict[fingerprint])
    
    return np.array(fingerprints)


def create_amino_acids(acids):
    retval = [acid_dict[acid_name] if acid_name in amino_list else acid_dict['MET'] if acid_name=='FME' else acid_dict['TMP'] for acid_name in acids]
    retval = np.array(retval)
    
    return(np.array(retval))


def dump_dictionary(dictionary, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def is_empty_pdb(pdb_id):
    
    empty = False
    
    with open(pdb_id + '.pdb', 'r') as f:
        for ln in f:
            if ln.startswith('<html>'):
                empty = True
                break
    return empty

def replace_pdb(pdb_id):
    with open(pdb_id + '.pdb', 'r') as f:
        filedata = f.read()
        filedata = filedata.replace('DBREF1', 'DBREF')
        filedata = filedata.replace('DBREF2', 'DBREF')
        filedata = filedata.replace('DBREF3', 'DBREF')
        filedata = filedata.replace('DBREF4', 'DBREF')
        filedata = filedata.replace('DBREF5', 'DBREF')
        filedata = filedata.replace('\\\'', 'P')
        
    with open(pdb_id + '.pdb', 'w') as f:
        f.write(filedata)

def parse_PDB(pdb_name, uniprot_id, user_chain):
    
    without_chain = False
    
    try:
        if not user_chain == '0':
            for record in SeqIO.parse(pdb_name+".pdb", "pdb-seqres"):
                pdb_id = record.id.strip().split(':')[0]
                chain  = record.annotations["chain"]
                _, UNP_id = record.dbxrefs[0].strip().split(':')

                if UNP_id == uniprot_id:
                    chain = record.annotations["chain"]
                    if chain == user_chain:
                        break

            if not chain:
                chain = user_chain
        else:
            chain = user_chain
            without_chain = True
    except:
        chain = user_chain
    
    with open(pdb_name+".pdb","r") as fi:
        mdl = False
        for ln in fi:
            if ln.startswith("NUMMDL"):
                mdl = True
                break
    
    with open(pdb_name+".pdb","r") as fi:
        id = []
        
        if mdl:
            for ln in fi:
                if ln.startswith("ATOM") or ln.startswith("HETATM"):
                    id.append(ln)
                elif ln.startswith("ENDMDL"):
                    break
        else:
            for ln in fi:
                if ln.startswith("ATOM") or ln.startswith("HETATM"):
                    id.append(ln)

    count = 0
    seq   = {}
    seq['type_atm'], seq['ind'], seq['amino'], seq['group'], seq['coords'] = [], [], [], [], []
    

    for element in id:
        type_atm = element[0:6].strip().split()[0]
        ind      = int(element[6:12].strip().split()[0])
        atom     = element[12:17].strip().split()[0]
        amino    = element[17:21].strip().split()[0]
        chain_id = element[21]
        group_id = int(element[22:26].strip().split()[0])
        x_coord  = float(element[30:38].strip().split()[0])
        y_coord  = float(element[38:46].strip().split()[0])
        z_coord  = float(element[46:54].strip().split()[0])
        
        coords   = np.array([x_coord, y_coord, z_coord])
        
        if not without_chain:
            if chain_id == chain:
                seq['type_atm'].append(type_atm)
                seq['ind'].append(int(ind))
                seq['amino'].append(amino)
                seq['group'].append(int(group_id))
                seq['coords'].append(coords)

                count += 1
        else:
            seq['type_atm'].append(type_atm)
            seq['ind'].append(int(ind))
            seq['amino'].append(amino)
            seq['group'].append(int(group_id))
            seq['coords'].append(coords)

            count += 1
    
    return seq['type_atm'], seq['amino'], seq['group'], seq['coords'], chain


def unique(list1): 
    x = np.array(list1) 
    return np.unique(x)


def group_by_coords(group,amino,coords):
    uniq_group   = np.unique(group)
    group_coords = np.zeros((uniq_group.shape[0],3))
    
    group_amino  = []
    
    np_group     = np.array(group)
    
    for i,e in enumerate(uniq_group):
        inds = np.where(np_group==e)[0]
        group_coords[i,:] = np.mean(np.array(coords)[inds],axis=0)
        group_amino.append(amino[inds[0]])
    
    return group_coords, group_amino

def get_graph_from_struct(group_coords, group_amino):
    num_residues = group_coords.shape[0]
    
    if (num_residues > max_residues):
        num_residues = max_residues
    
    residues = group_amino[:num_residues]
        
    retval = [ [0 for i in range(0, num_residues)] for j in range(0, num_residues)]
    
    residue_type = []
    for i in range(0, num_residues):
        if residues[i] == 'FME':
            residues[i] = 'MET'
        elif residues[i] not in amino_list:
            residues[i] = 'TMP'
            
        residue_type.append(residues[i])
        
        for j in range(i+1, num_residues):
            x, y = group_coords[i], group_coords[j]
            retval[i][j] = LA.norm(x-y)
            retval[j][i] = retval[i][j]
    
    retval = np.array(retval)
    
    threshold = 9.5
    
    for i in range(0, num_residues):
        for j in range(0, num_residues):
            if (retval[i,j] <= threshold):
                retval[i,j] = 1
            else:
                retval[i,j] = 0
    
    n          = retval.shape[0]
    adjacency  = retval + np.eye(n)
    degree     = sum(adjacency)
    d_half     = np.sqrt(np.diag(degree))
    d_half_inv = np.linalg.inv(d_half)
    adjacency  = np.matmul(d_half_inv,np.matmul(adjacency,d_half_inv))
    return residue_type, np.array(adjacency)



with open('list_of_prots.txt', 'r') as f:
    data_list = f.read().strip().split('\n')

pdb_to_uniprot_dict, pdb_to_chain_dict = {}, {}
for data in data_list:
    pdb = data.strip().split('\t')[1]
    pdb_to_uniprot_dict[pdb] = data.strip().split('\t')[0]
    pdb_to_chain_dict[pdb] = data.strip().split('\t')[2]
    


radius = 1

atom_dict = defaultdict(lambda: len(atom_dict))
acid_dict = defaultdict(lambda: len(acid_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

filepath = "pdb_files/"
os.chdir(filepath)

dir_input = ('input'+str(radius)+'/')
os.makedirs(dir_input, exist_ok=True)

os.chdir('../')

f = []
for (dirpath, dirnames, filenames) in walk(filepath):
    f.extend(filenames)
    break

os.chdir(filepath)

pdb_ids = []
for data in f:
    pdb_ids.append(data.strip().split('.')[0])
    
num_prots = len(pdb_ids)

count   = 0
q_count = 0

adjacencies, proteins, pnames, pseqs = [], [], [], []


for n in range(num_prots):    
    if not is_empty_pdb(pdb_ids[n]):
        replace_pdb(pdb_ids[n])
        pdb_name   = pdb_ids[n]
        
        uniprot_id = pdb_to_uniprot_dict[pdb_ids[n]]
        user_chain = pdb_to_chain_dict[pdb_ids[n]]
        
        print('/'.join(map(str, [n+1, num_prots])))
        print(pdb_name)
        
        try:
            type_atm, amino, group, coords, chain = parse_PDB(pdb_name, uniprot_id, user_chain)
            group_coords, group_amino             = group_by_coords(group,amino,coords)
            residue_type, adjacency               = get_graph_from_struct(group_coords, group_amino)
            atoms                                 = create_amino_acids(residue_type)
            
            fingerprints = create_fingerprints(atoms, adjacency, radius)
            
            adjacencies.append(adjacency)
            proteins.append(fingerprints)
            pnames.append(pdb_name)
            
            d_seq = {}
            for no, g in enumerate(group):
                if g not in d_seq.keys():
                    d_seq[g] = amino[no]
            
            seq_pr = ''
            for k in d_seq:
                if d_seq[k] in amino_list:
                    seq_pr += amino_short[d_seq[k]]
            
            pseqs.append(seq_pr)
            
            count += 1
            
            if count%10 == 0 or n==num_prots-1:
                count    = 0
                
                np.save(dir_input + 'proteins_' + str(10*q_count+1) + '_' + str(10*(q_count+1)), proteins)
                np.save(dir_input + 'adjacencies_' + str(10*q_count+1) + '_' + str(10*(q_count+1)), adjacencies)
                np.save(dir_input + 'names_' + str(10*q_count+1) + '_' + str(10*(q_count+1)), pnames)
                np.save(dir_input + 'seqs_' + str(10*q_count+1) + '_' + str(10*(q_count+1)), pseqs)
                
                adjacencies, proteins, pnames, pseqs = [], [], [], []
                q_count += 1
            
        except:
            print(pdb_name, uniprot_id, user_chain)
            
dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
print('Length of fingerprint dictionary: '+str(len(fingerprint_dict)))