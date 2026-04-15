import re
import numpy as np
from scipy.stats import fisher_exact as f
import pandas as pd
from math import comb
from Bio import SeqIO


def preprocess(any_df):
    """
    Preprocesses UniProt SPARQL Query with specific column transformations and cleaning.

    Parameters
    ----------
    any_df : DataFrame
        Input dataframe containing uniprot query outputs.

    Returns
    -------
    DataFrame
        Preprocessed dataframe.
    """
    
    any_df = any_df.astype(str)
    
    # rename columns
    any_df.columns = ['ID', 'name', 'fullName', 'substrate', 'location', 'moiety', 'AminoAcid', 'position',
                      'evidenceCode', 'publication', 'ProteinEntryReviewed']

    # In column Name remove string _HUMAN
    for i, name in enumerate(any_df['name']):
        any_df['name'][i] = re.sub('_HUMAN',  '', name)

    # clean column substrate
    pat1 = r'^(.*superfamily. )'
    pat2 = r'Belongs to the '
    pat3 = r' family.*$'
    combined_pat = r'|'.join((pat1, pat2, pat3))

    # clean addition after ; for moiety
    for i, m in enumerate(any_df['moiety']):
        any_df['moiety'][i] = re.sub(';.*', '', m)

    # column position, remove everything after first two characters
    for i, position in enumerate(any_df['position']):
        any_df['position'][i] = position[:2]

    # column ProteinEntryReviewed remove string after true or false
    for i, per in enumerate(any_df['ProteinEntryReviewed']):
        any_df['ProteinEntryReviewed'][i] = (re.sub("\^\^<http://www.w3.org/2001/XMLSchema#boolean>",  '', per))

    # transform all locations containing "membrane" to just membrane
    for i, location in enumerate(any_df['location']):
        if 'membrane' in location.lower():
            any_df['location'][i] = 'Membrane'

    # transform nan strings to true NaN values
    any_df = any_df.replace('nan', np.NaN)
    any_df = any_df.replace('na', np.NaN)
    
    for i, sub in enumerate(any_df['substrate']):
        if pd.isna(sub):
            any_df['substrate'][i] = any_df['name'][i]
            pass
        else:
            any_df['substrate'][i] = re.sub(combined_pat, '', sub)
            
    # make sure that the position entries are integers, not strings
    any_df['position'] = any_df['position'].fillna(0).astype(int)

    # remove double entries
    any_df = any_df.drop_duplicates()
    any_df = any_df.reset_index(drop=True)

    return any_df


def preprocess_background(any_df):
    """
    Preprocesses the UniProt SPARQL Query output of background proteins with specific column
    transformations and cleaning.

    Parameters
    ----------
    any_df : DataFrame
        Input dataframe containing uniprot query outputs for background proteins.

    Returns
    -------
    DataFrame
        Preprocessed background dataframe.
    """
    
    any_df = any_df.astype(str)

    # rename columns
    any_df.columns = ['ID', 'location', 'ProteinEntryReviewed']

    # column ProteinEntryReviewed remove string after true or false
    for i, per in enumerate(any_df['ProteinEntryReviewed']):
        any_df['ProteinEntryReviewed'][i] = (re.sub("\^\^<http://www.w3.org/2001/XMLSchema#boolean>",  '', per))

    # transform all locations containing "membrane" to just membrane
    for i, location in enumerate(any_df['location']):
        if 'membrane' in location.lower():
            any_df['location'][i] = 'Membrane'

    # transform nan strings to true NaN values
    any_df = any_df.replace('nan', np.NaN)
    any_df = any_df.replace('na', np.NaN)

    # remove double entries
    any_df = any_df.drop_duplicates()
    any_df = any_df.reset_index(drop=True)

    return any_df


def percent(part, whole):
    """
    Computes the percentage based on the given part and whole.

    Parameters
    ----------
    part : float
        Numerator value.
    whole : float
        Denominator value.

    Returns
    -------
    float
        Calculated percentage.
    """
    
    try:
        return 100 * (float(part) / float(whole))
    except ZeroDivisionError:
        return 0


def clean_agg(any_df):
    """
    Cleans the given aggregated dataframe by transforming specific columns.

    Parameters
    ----------
    any_df : DataFrame
        Input aggregated dataframe to be cleaned.

    Returns
    -------
    DataFrame
        Cleaned aggregated dataframe.
    """

    columns = list(any_df)
    del columns[0]

    for col in columns:
        for i, elem in enumerate(any_df[col]):
            any_df[col][i] = list(elem)
            if len(elem) <= 1:
                elem = list(elem)
                any_df[col][i] = elem[0]
                
    if 'location' in any_df.columns:
        for i, x in enumerate(any_df['location']):
            if isinstance(x, list):
                if 'Membrane' in x:
                    any_df['location'][i] = 'Membrane'

    if 'position' in any_df.columns:
        # sort positions, so 1, 2 are regarded as the same as 2, 1
        for i, pos in enumerate(any_df['position']):
            if isinstance(pos, list):
                any_df['position'][i] = sorted(pos)

    for col in columns:
        for i, elem in enumerate(any_df[col]):
            if isinstance(elem, list):
                any_df[col][i] =  [x for x in elem if not isinstance(x, float)]

    for col in columns:
        for i, elem in enumerate(any_df[col]):
            if isinstance(elem, list):
                if len(elem) < 1:
                    elem = list(elem)
                    any_df[col][i] = np.NaN                
                
    return any_df


def add_motif(any_df):
    """
    Adds a motif column to the given dataframe based on position values.

    Parameters
    ----------
    any_df : DataFrame
        Input dataframe with position information.

    Returns
    -------
    DataFrame
        DataFrame with added motif column.
    """
    
    # add column for motif for better readability of 'position'
    
    any_df['motif'] = 0
    
    for i, pos in enumerate(any_df['position']):
        if pos == -4:
            any_df.at[i, 'motif'] = 'CXXX'

        if pos == [-3, -1]:
            any_df.at[i, 'motif'] = 'CXC'

        if pos == [-2, -1]:
            any_df.at[i, 'motif'] = 'CC'

        if pos == [-4, -3]:
            any_df.at[i, 'motif'] = 'CCXX'

        if pos == [-5, -4]:
            any_df.at[i, 'motif'] = 'CCXXX'

        if pos == [-3, -2]:
            any_df.at[i, 'motif'] = 'CCX'
            
        if pos == -1:
            any_df.at[i, 'motif'] = 'C'
     
    return any_df
                

def pep_Cterm(df):
    """
    Adds peptides for all C-terminal C positions to the given dataframe.

    Parameters
    ----------
    df : DataFrame
        Input dataframe containing protein information.

    Returns
    -------
    DataFrame
        DataFrame with added C-terminal peptide column.
    """

    df['pep'] = df.seq.str[-10:]

    return df


def pep_intern(df):
    """
    Adds 11 AA long peptides for internal C positions to the given DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing protein information with 'seq' and 'Cpos' columns.
        
    Returns
    -------
    DataFrame
        A new DataFrame with an added 'pep' column containing peptides.
    """
    # Ensure 'seq' and 'Cpos' exist in the DataFrame
    if 'seq' not in df.columns or 'Cpos' not in df.columns:
        raise ValueError("DataFrame must contain 'seq' and 'Cpos' columns.")
    
    # Generate peptides
    def generate_peptide(row):
        # Convert Cpos (negative index) to a positive index
        pos = len(row['seq']) + int(row['Cpos'])
        
        # Calculate the start and end for the slice
        start = max(0, pos - 5)  # Prevent start index from going negative
        end = min(len(row['seq']), pos + 6)  # Prevent end index from exceeding the sequence length
        
        return row['seq'][start:end]
    
    # Apply function to generate peptides
    df['pep'] = df.apply(generate_peptide, axis=1)
    
    return df


def annotate_pep_internal(any_df):
    """
    Annotates internal peptide positions based on specific motifs (C/CC/CXC/CCC) in the given dataframe.

    Parameters
    ----------
    any_df : DataFrame
        Input dataframe with peptide information.

    Returns
    -------
    DataFrame
        DataFrame with annotated peptide positions for internal motifs.
    """
    
    # match motif on peptides

    motif_patterns = {
        'pepCCC': r'[A-Z]{3}C{3}[A-Z]{3}', # 3 any - 3 C - 3 any
        'pepCC': r'[A-Z]{3}[^C]{1}C{2}[^C]{1}[A-Z]{3}', # 3 any - 1 not C - 2 C - 1 not C - 4 any
        'pepCXC': r'[A-Z]{2}[^C]{1}C{1}[^C]{1}C{1}[^C]{1}[A-Z]{2}', # 2 any - 1 not C - 1 C - 1 not C - 1 C - 1 not C - 2 any
        'pepC': r'[A-Z]{3}[^C]{2}C{1}[^C]{2}[A-Z]{3}', # 3 any - 2 not C - 1 C - 2 not C - 3 any
    }


    for motif_name, pattern in motif_patterns.items():
        # match motif pattern
        any_df[motif_name] = any_df['pep'].str.findall(pattern)
        # replace empty lists with true NaN values
        any_df.loc[any_df[motif_name].str.len() == 0, motif_name] = np.nan
        # unpack from lists
        mask = any_df[motif_name].str.len() == 1
        any_df.loc[mask, motif_name] = [any_df.loc[mask, motif_name][i][0] for i in any_df.loc[mask, motif_name].index]
    
    return any_df


def update_fasta(fasta, index, new_value):
    """
    Update the 'Cpos' value in the given 'fasta' DataFrame at the specified 'index' with the 'new_value'.

    Parameters:
        fasta (pandas.DataFrame): The DataFrame containing the sequence data with 'Cpos' column.
        index (int): The index of the row where the 'Cpos' value should be updated.
        new_value: The new value to be assigned to the 'Cpos' column.

    Returns:
        None. The 'Cpos' value in the 'fasta' DataFrame at the specified 'index' is updated with 'new_value'.
    """

    fasta.at[index, 'Cpos'] = new_value


class Urn:

    def __init__(self, K_arr):
        """
        Initialization given the number of each type i object in the urn.

        Parameters
        ----------
        K_arr: ndarray(int)
            number of each type i object.
        """

        self.K_arr = np.array(K_arr)
        self.N = np.sum(K_arr)
        self.c = len(K_arr)

    def pmf(self, k_arr):
        """
        Probability mass function.

        Parameters
        ----------
        k_arr: ndarray(int)
            number of observed successes of each object.
        """

        K_arr, N = self.K_arr, self.N

        k_arr = np.atleast_2d(k_arr)
        n = np.sum(k_arr, 1)

        num = np.prod(comb(K_arr, k_arr), 1)
        denom = comb(N, n)

        pr = num / denom

        return pr

    def moments(self, n):
        """
        Compute the mean and variance-covariance matrix for
        multivariate hypergeometric distribution.

        Parameters
        ----------
        n: int
            number of draws.
        """

        K_arr, N, c = self.K_arr, self.N, self.c

        # mean
        m = n * K_arr / N

        # variance-covariance matrix
        v = np.full((c, c), n * (N - n) / (N - 1) / N ** 2)
        for i in range(c-1):
            v[i, i] *= K_arr[i] * (N - K_arr[i])
            for j in range(i+1, c):
                v[i, j] *= - K_arr[i] * K_arr[j]
                v[j, i] = v[i, j]

        v[-1, -1] *= K_arr[-1] * (N - K_arr[-1])

        return m, v

    def simulate(self, n, size=1, seed=None):
        """
        Simulate a sample from multivariate hypergeometric
        distribution where at each draw we take n objects
        from the urn without replacement.

        Parameters
        ----------
        n: int
            number of objects for each draw.
        size: int(optional)
            sample size.
        seed: int(optional)
            random seed.
        """

        K_arr = self.K_arr

        gen = np.random.Generator(np.random.PCG64(seed))
        sample = gen.multivariate_hypergeometric(K_arr, n, size=size)

        return sample
    
# from https://python.quantecon.org/multi_hyper.html, 22.05.2023


def fishers_exact_test(x, N, n, M):
    """
    Performs Fisher's exact test for enrichment analysis.

    Parameters
    ----------
    x : int
        Number of successful outcomes.
    N : int
        Total number of outcomes.
    n : int
        Number of successful outcomes in the reference set.
    M : int
        Total number of outcomes in the reference set.

    Returns
    -------
    float
        p-value from the Fisher's exact test.
    """
    
    table = np.array([[x, n-x],
                      [N-x, M - (n + N) + x]
                      ])
    table[table < 0] = 0
        
    oddsr, p = f(table, alternative='two-sided')
    
    return p


def add_Cpos(fasta_df):
    """
    Adds C positions counting from N-terminal and counting from C-terminal to the given dataframe based on protein sequences.

    Parameters
    ----------
    fasta_df : DataFrame
        Input dataframe containing protein sequences.

    Returns
    -------
    DataFrame
        DataFrame with added C position columns.
    """
    
    allC = []
    N_Cpos = []
    
    for i in range(len(fasta_df)):
        for pos, amino in enumerate((fasta_df.iloc[i]['seq'])):
            if amino == 'C':
                N_Cpos.append([fasta_df.iloc[i]['ID'], pos+1])
                allC.append(-(fasta_df.iloc[i]['len'] - (pos))) 
    
    N_Cpos = pd.DataFrame(N_Cpos).rename(columns={0: 'ID', 1: 'N_Cpos'})
    fasta_df = fasta_df.merge(N_Cpos, on='ID')
    
    fasta_df.insert(4, 'Cpos', allC)
    
    return fasta_df
    
    
def get_Ccount(singleCs_df):
    """
    Calculate the count of C residues in proteins by aggregating the data.

    Parameters:
        singleCs_df (pd.DataFrame): DataFrame containing information about C residues.

    Returns:
        pd.DataFrame: DataFrame with protein ID, C count, and other relevant information.
    """
    
    df = singleCs_df.groupby('ID').agg(set).reset_index()
    df = clean_agg(df)
    
    df_single = pd.DataFrame([df.iloc[i] for i, x in enumerate(df['N_Cpos']) if not isinstance(x, list)]).reset_index(drop=True)
    df_double = pd.DataFrame([df.iloc[i] for i, x in enumerate(df['N_Cpos']) if isinstance(x, list)]).reset_index(drop=True)
    
    # sort lists of positions
    df_double['N_Cpos'] = [sorted(x) for x in df_double['N_Cpos']]

    df_single['Ccount'] = 1
    df_double['Ccount'] = [len(df_double['N_Cpos'][i]) for i, row in enumerate(df_double.values)]

    df_Ccount = pd.concat([df_single, df_double]).sort_values('Ccount').reset_index(drop=True)
    
    if 'seq' in df_Ccount.columns:
        df_Ccount['Count_all'] = [df_Ccount['seq'][i].count('C') for i, x in enumerate(df_Ccount.values)]
    
    return df_Ccount


def c_term_pep(peptides):
    """
    Extract the most C-terminal peptides from detected proteins.

    Parameters:
        peptides (pd.DataFrame): DataFrame containing peptide information.

    Returns:
        pd.DataFrame: DataFrame with the most C-terminal peptides for each protein.
    """
    
    # reduce variables in TE
    peptides = peptides.loc[:, ['Sequence', 'ID', 'Start_position', 'End_position']]

    # Getting the most C-terminal peptide (with the highest end position) of a detected protein
    peptides = peptides.loc[peptides.groupby(["ID"])["End_position"].idxmax()].reset_index(drop=True)
    peptides = peptides.iloc[:, 0:4]
    peptides = peptides.sort_values(by='ID')
    
    return peptides
    

def peptides_end_pos(peptides_end, fastafile):
    """
    Calculate and annotate peptide end positions based on protein lengths.

    Parameters:
        peptides_end (pd.DataFrame): DataFrame containing peptide end positions.
        fastafile (pd.DataFrame): DataFrame containing protein sequence information.

    Returns:
        pd.DataFrame: DataFrame with annotated peptide end positions.
    """
    
    # Add a column for the protein lenght from fasta file
    peptides_end = pd.merge(peptides_end, fastafile, on='ID', how='inner')
    
    # Add start_pos and end_pos columns for position counting from C-teminal backwards
    start_pos = []
    end_pos = []

    for i, start in enumerate(peptides_end['Start_position']):
        start_pos.append(int(-((peptides_end['len'][i] - start) + 1)))

    for i, end in enumerate(peptides_end['End_position']):
        end_pos.append(int(-((peptides_end['len'][i] - end) + 1)))

    peptides_end.insert(4, 'start_pos', start_pos)
    peptides_end.insert(5, 'end_pos', end_pos)
    
    # annotate the range of the peptides
    peptides_end['pep_range'] = peptides_end.apply(lambda x: list(range(x['start_pos'], x['end_pos']+1)), 1)
    
    return peptides_end
    

def pep_bin_count(peptides_end):
    """
    Count the occurrence of peptide positions within predefined bins.

    Parameters:
        peptides_end (pd.DataFrame): DataFrame containing annotated peptide end positions.

    Returns:
        pd.DataFrame: DataFrame with counts of peptide positions in predefined bins.
    """
    
    # create bins for barplot of peptide positions
    bins = pd.IntervalIndex.from_tuples([(-300, -291), (-290, -281), (-280, -271), (-270, -261), (-260, -251), (-250, -241), 
                                        (-240, -231), (-230, -221), (-220, -211), (-210, -201), (-200, -191), (-190, -181),
                                        (-180, -171), (-170, -161), (-160, -151), (-150, -141), (-140, -131), (-130, -121),
                                        (-120, -111), (-110, -101), (-100, -91), (-90, -81), (-80, -71), (-70, -61), (-60, -51),
                                        (-50, -41), (-40, -31), (-30, -21), (-20, -11), (-10, -2), (-1, -1)], closed='both')
    
    # create labels for each bin
    labels = []
    for c in bins:
        l = ' '.join([str(c.left), "to", str(c.right)])
        labels.append(l)
    labels[-1] = '-1'

    # Put the peptide positions into bins, keeping only one count per bin
    pep_bins = []
    for i, row in enumerate(peptides_end.values):
        x = pd.cut(peptides_end['pep_range'][i], bins).unique()
        x = pd.Series(x).astype('str') 
        pep_bins.append(x)

    pep_bins = pd.DataFrame(np.array(pep_bins, dtype=object), columns=['bins'])

    # Count the occurence of bins per dataframe, for use in barplot
    counter = {'[-300, -291]': 0, '[-290, -281]': 0, '[-280, -271]': 0, '[-270, -261]': 0, '[-260, -251]': 0, '[-250, -241]': 0, 
               '[-240, -231]': 0, '[-230, -221]': 0, '[-220, -211]': 0, '[-210, -201]': 0, '[-200, -191]': 0, '[-190, -181]': 0, 
               '[-180, -171]': 0, '[-170, -161]': 0, '[-160, -151]': 0, '[-150, -141]': 0, '[-140, -131]': 0, '[-130, -121]': 0, 
               '[-120, -111]': 0, '[-110, -101]': 0, '[-100, -91]': 0, '[-90, -81]': 0, '[-80, -71]': 0, '[-70, -61]': 0, 
               '[-60, -51]': 0, '[-50, -41]': 0, '[-40, -31]': 0, '[-30, -21]': 0, '[-20, -11]': 0, '[-10, -2]': 0, '[-1, -1]': 0}

    for i, row in enumerate(pep_bins.values):
        for x in row:
            for y in x:
                if y not in counter:
                    counter[y] = 0
                counter[y] += 1

    # store in df for barplot
    pep_pos = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    pep_pos = pep_pos.rename(columns={0: "y"})
    pep_pos = pep_pos.loc[pep_pos["index"] != 'nan']
    pep_pos['x'] = labels
    
    return pep_pos


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
    

def pep_table(dfCI, dfTE, fasta):
    """
    Generate a table of peptides with additional annotations based on click-it and total extract data.

    Parameters:
        dfCI (pd.DataFrame): DataFrame containing click-it peptide data.
        dfTE (pd.DataFrame): DataFrame containing total extract peptide data.
        fasta (pd.DataFrame): DataFrame containing protein sequence information.

    Returns:
        pd.DataFrame: DataFrame with annotated peptide information.
    """
    
    # rename and merge click-it and total extract on protein
    dfCI.rename(columns={'Sequence': 'sequence_CI', 'end_pos': 'end_pos_CI'}, inplace=True)
    df_CI = dfCI[['sequence_CI', 'ID', 'end_pos_CI']]

    dfTE.rename(columns={'Sequence': 'sequence_TE', 'start_pos': 'start_pos_TE', 'end_pos': 'end_pos_TE'},
                inplace=True)
    df_TE = dfTE[['sequence_TE', 'ID', 'start_pos_TE', 'end_pos_TE']]

    df = pd.merge(df_CI, df_TE, on='ID', how='outer')
    
    # Filter df where start_pos_TE is bigger than end_pos_CI, meaning that the total extract contains
    # additional peptides compared to the click-it data
    df = df[(df['start_pos_TE'] - df['end_pos_CI']) > 1].reset_index(drop=True)
    
    # Add the complete fasta sequence to the df
    df = pd.merge(df, fasta[['seq', 'ID']], on='ID', how='inner')
    
    # which proteins have at least one C anywhere after the CI peptide?
    df_table = []

    for i, x in enumerate(df.values):
        if x[6][int(df['end_pos_CI'][i]): len(x[6])].count('C') > 0:
            df_table.append(x)

    df_table = pd.DataFrame(df_table, columns=['sequence_CI', 'ID', 'end_pos_CI', 'sequence_TE',
                                               'start_pos_TE', 'end_pos_TE', 'seq'])
    
    return df_table


def pep_table(dfCI, dfTE, fasta):
    """
    Generate a table of peptides with additional annotations based on click-it and total extract data.

    Parameters:
        dfCI (pd.DataFrame): DataFrame containing click-it peptide data.
        dfTE (pd.DataFrame): DataFrame containing total extract peptide data.
        fasta (pd.DataFrame): DataFrame containing protein sequence information.

    Returns:
        pd.DataFrame: DataFrame with annotated peptide information.
    """
    
    # rename and merge click-it and total extract on protein
    dfCI.rename(columns={'Sequence': 'sequence_CI', 'end_pos': 'end_pos_CI'}, inplace=True)
    df_CI = dfCI[['sequence_CI', 'ID', 'end_pos_CI']]

    dfTE.rename(columns={'Sequence': 'sequence_TE', 'start_pos': 'start_pos_TE', 'end_pos': 'end_pos_TE'},
                inplace=True)
    df_TE = dfTE[['sequence_TE', 'ID', 'start_pos_TE', 'end_pos_TE']]

    df = pd.merge(df_CI, df_TE, on='ID', how='outer')
    
    # Filter df where start_pos_TE is bigger than end_pos_CI, meaning that the total extract contains
    # additional peptides compared to the click-it data
    df = df[(df['start_pos_TE'] - df['end_pos_CI']) > 1].reset_index(drop=True)
    
    # Add the complete fasta sequence to the df
    df = pd.merge(df, fasta[['seq', 'ID']], on='ID', how='inner')
    
    # which proteins have at least one C anywhere after the CI peptide?
    df_table = []

    for i, x in enumerate(df.values):
        if x[6][int(df['end_pos_CI'][i]): len(x[6])].count('C') > 0:
            df_table.append(x)

    df_table = pd.DataFrame(df_table, columns=['sequence_CI', 'ID', 'end_pos_CI', 'sequence_TE',
                                               'start_pos_TE', 'end_pos_TE', 'seq'])
    
    return df_table


def pep_table_reverse(dfCI, dfTE, fasta):
    """
    Generate a table of peptides with additional annotations based on click-it and total extract data.

    Parameters:
        dfCI (pd.DataFrame): DataFrame containing click-it peptide data.
        dfTE (pd.DataFrame): DataFrame containing total extract peptide data.
        fasta (pd.DataFrame): DataFrame containing protein sequence information.

    Returns:
        pd.DataFrame: DataFrame with annotated peptide information.
    """
    
    # rename and merge click-it and total extract on protein
    dfCI.rename(columns={'Sequence': 'sequence_CI', 'end_pos': 'end_pos_CI'}, inplace=True)
    df_CI = dfCI[['sequence_CI', 'ID', 'end_pos_CI']]

    dfTE.rename(columns={'Sequence': 'sequence_TE', 'start_pos': 'start_pos_TE', 'end_pos': 'end_pos_TE'},
                inplace=True)
    df_TE = dfTE[['sequence_TE', 'ID', 'start_pos_TE', 'end_pos_TE']]

    df = pd.merge(df_CI, df_TE, on='ID', how='outer')
    
    # Filter df where start_pos_TE is bigger than end_pos_CI, meaning that the total extract contains
    # additional peptides compared to the click-it data
    df = df[(df['start_pos_TE'] - df['end_pos_CI']) < 1].reset_index(drop=True)
    
    # Add the complete fasta sequence to the df
    df = pd.merge(df, fasta[['seq', 'ID']], on='ID', how='inner')

    df_table = pd.DataFrame(df, columns=['sequence_CI', 'ID', 'end_pos_CI', 'sequence_TE',
                                               'start_pos_TE', 'end_pos_TE'])
    
    return df_table


def known_new_motifs_agg(dataframe):
    """
    Aggregate peptide data and separate proteins with known and new motifs.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing peptide data.

    Returns:
        pd.DataFrame, pd.DataFrame: Two DataFrames containing aggregated peptide data for known and new motifs.
    """
    
    agg_dataframe = dataframe.groupby('ID').agg(set).reset_index()
    agg_dataframe = clean_agg(agg_dataframe)
    
    filter_condition = (agg_dataframe['pepCAAX'].isnull() &
                        agg_dataframe['pepCXXX'].isnull() &
                        agg_dataframe['pepCC'].isnull() &
                        agg_dataframe['pepCXC'].isnull() &
                        agg_dataframe['pepCCX'].isnull() &
                        agg_dataframe['pepCCXX'].isnull() &
                        agg_dataframe['pepCCXXX'].isnull())
    
    other_dataframe = agg_dataframe[filter_condition].reset_index(drop=True)
    
    return agg_dataframe, other_dataframe


def annotate_most_accs_C(intern_pep):
    """
    Annotate most accessible C residues based on structural data.

    Parameters:
        intern_pep (pd.DataFrame): DataFrame containing peptide and protein data.

    Returns:
        pd.DataFrame: DataFrame with annotated information about most accessible C residues.
    """
    
    intern_accs = intern_pep[['ID', 'nAA_30_180_pae']]
    intern_accs = intern_accs.groupby('ID').agg(set).reset_index()
    
    # Clean aggregated data and rename the column
    intern_accs = clean_agg(intern_accs).rename(columns={'nAA_30_180_pae': 'lowest_pae'})
    
    # Apply a lambda function to keep only the lowest number in each list, skipping if it's already an integer
    intern_accs['lowest_pae'] = intern_accs['lowest_pae'].apply(lambda x: min(x) if isinstance(x, list) else x)
    
    # Merge the two DataFrames based on the 'ID' column.
    merged_df = pd.merge(intern_pep, intern_accs, on="ID", suffixes=("_pep", "_accs"))

    # Filter rows where nAA_30_180_pae equals lowest_pae from intern_accs.
    lowest_pae_inter_pep = merged_df[merged_df["nAA_30_180_pae"] == merged_df["lowest_pae"]].reset_index(drop=True)

    # Aggregate the Data to find counts of most accessible C residues for each protein ID
    agg_intern_accs = clean_agg(lowest_pae_inter_pep.groupby('ID').agg(set).reset_index())

    # Separate single-accessible and double-accessible residues into separate DataFrames
    agg_intern_accs_single = pd.DataFrame([agg_intern_accs.iloc[i] for i, x in enumerate(agg_intern_accs['Cpos']) if not isinstance(x, list)]).reset_index(drop=True)
    agg_intern_accs_double = pd.DataFrame([agg_intern_accs.iloc[i] for i, x in enumerate(agg_intern_accs['Cpos']) if isinstance(x, list)]).reset_index(drop=True)

    # Annotate accessible residue counts
    agg_intern_accs_single['accs_count'] = 1
    agg_intern_accs_double['accs_count'] = [len(x) for x in agg_intern_accs_double['Cpos']]

    # Combine both dataframes back and sort by accessible counts
    agg_intern_accs = pd.concat([agg_intern_accs_single, agg_intern_accs_double]).sort_values('accs_count').reset_index(drop=True)

    return agg_intern_accs
