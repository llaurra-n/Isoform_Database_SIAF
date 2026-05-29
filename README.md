# Proteoform-Specific Search Databases for Prenylated Isoforms

## Project with SIAF and ETH Studios Davos

An engineered, enzyme-specific database strategy designed to accurately identify alternative splice isoforms and post-translationally modified proteoforms without the search-space penalties of traditional redundant databases.

## Overview

Traditional proteomic search pipelines struggle to identify unique alternative splice isoforms and specific proteoforms. Including full-length redundant sequences inflates the search space, leading to strict False Discovery Rate (FDR) penalties that routinely wash out novel identifications.

This project scales a targeted strategy to bypass these limitations. By isolating and targeting **only the unique sequence regions**, **alternative C-terminal variants**, and **neo-N-termini** generated via post-prenylation processing, this custom database restricts the search space strictly to enzyme-specific, unique proteoform regions. 

The resulting streamlined databases can be integrated directly into proteotypic-filtered search pipelines like MaxQuant.

---

## Proof of Concept

The feasibility of this targeted approach was demonstrated in a pilot study of 34 prenylated proteins. These targets lacked canonical C-terminal prenylation motifs or target cysteines in their reference canonical forms, but contained them within alternative splice isoforms.

### Workflow & Results:
1. **In Silico Digestion:** Alternative C-terminal isoforms were computationally digested to extract isoform-specific C-terminal peptides.
2. **Unique Sequence Generation:** Distinct, unique sequences of sufficient length were successfully generated for **31 out of the 34 proteins**.
3. **Novel Discovery:** Appending this miniature, isoform-specific database to the reference human proteome and re-searching published Th1 cell mass spectrometry data resulted in the novel identification of a peptide unique to the third isoform of the signal peptidase complex subunit **SEC11A** (*Koch et al., 2025*).

---

## Project Objectives

Building on the pilot study, this repository contains the pipeline to scale this strategy comprehensively:

* **Enzyme-Specific Scalability:** Scale the targeted database creation into a comprehensive, automated pipeline.
* **C-Terminal Isolation:** Isolate unique alternative C-terminal variants.
* **Processing Integration:** Incorporate specific neo-N-terminal sequences generated through post-prenylation processing pathways.
* **Pipeline Integration:** Format and optimize the engineered databases for direct deployment in standard proteotypic-filtered search workflows.

---



