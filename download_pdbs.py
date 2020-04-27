# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:13:10 2020

@author: mayank
"""

import requests
import time
import os

def fetch_pdb_file(pdb_id):
    the_url = "https://files.rcsb.org/download/" + pdb_id
    page = requests.get(the_url)
    pdb_file = str(page.content)
    pdb_file = pdb_file.replace('\\n', '\n')
    return(pdb_file)
    
with open('list_of_prots.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    
pdb_list = []

for data in data_list:
    pdb_list.append(data.strip().split('\t')[1])

os.makedirs('pdb_files/', exist_ok=True)
ctr = 0
for pdb_id in pdb_list:
    pdbfile = fetch_pdb_file(pdb_id + ".pdb")
    filename = "pdb_files/" + pdb_id + ".pdb"
    print("Writing " + filename)
    with open(filename, "w") as fd:
        fd.write(pdbfile)
    ctr += 1
    if ctr%1000 == 0:
        time.sleep(60)

print("All done!")