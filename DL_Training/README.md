For the post-processing of normalized min-sum (NMS) decoding failures of short BCH codes such as (63,36), (63,39) or (63,45) codes, due to its block lengths, it is not neccessary to 
call in advanced ordered statistics decoding (OSD) as the one presented controlling computational complexity of decoding of LDPC codes in the grad-parent directiory. Thus the 
conventional OSD by Marc Fossorier as early as 1995 is called for simplity, despite the transfering of advanced OSD to the scenario is readily made.

This code module focuses on training a simple CNN model of double layers to boost the reliablity metric for the failed NMS sequences, in the hope of facilitating the succedded OSD decoding.
Compared with the decoding information aggreagation (DIA) model in LDPC codes,  they are actually alike in terms of network architecture, the mere difference is the DIA of  our short BCH code
is more shallow than in LDPC codes,  for the maximum iterations setting is only 4, rathter than the setting of 8 for the latter.

Noteably, in current module, the standard parity-check matrix is utilized throughout, while the optimized redundant parity-check matrix is called in iterative NMS decoding only.
thus in globalmap file:
    set_map('regular_matrix',True)
    set_map('generate_extended_parity_check_matrix',False)
