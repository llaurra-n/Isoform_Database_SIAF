import re
import pandas as pd
from Bio import SeqIO

def read_fastafile(fasta_filepath):
    """
    Read protein sequence information from a FASTA file.

    Parameters:
        fasta_filepath (str): Path to the FASTA file.

    Returns:
        pd.DataFrame: DataFrame containing protein sequence information.
    """
    
    fasta = []
    for seq_record in SeqIO.parse(fasta_filepath, "fasta"):
        fasta.append((seq_record.id, str(seq_record.seq), len(seq_record)))

    fasta = pd.DataFrame(fasta, columns=('seqID', 'seq', 'len'))

    # added ID column to match easily on other files
    ID = fasta['seqID'].str.split('|', expand=True)

    sequences = pd.concat([ID[1].rename('ID'), fasta], axis=1)

    return sequences
    

