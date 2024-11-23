# Short_BCH_Decoding_OSD
For the project, it adopts the architecture of NMS+DIA+OSD, holding the benefits of low-complexity low-latency,
high decoding performance and indepedence of noise variance estimation etc.
The related manuscript entitled 'Iterative Decoding of Short BCH Codes and its Post-processing' is:
https://arxiv.org/abs/2411.13876

The interactions among the involved modules:
Training route:
1)Training_data_gen_63 module generates training data file;
2)BCH_63_training module optimizes the only parameter of NMS and generate training data file for DIA model.
3)DL_Training module uses the output file  of step 2 to train DIA model.
Testing route:
4)Tesing_data_gen_63 module generates testing data files at varied SNR points.
5)BCH_63_testing module generates testing results for output files in step 4 and generate varied files with
NMS decoding failures included.
6)DL_OSD_Testing module utilzes trained DIA model of step 3 and post-processes decoding failure files of 
step 5 using ordered statistics decoding.

Notice: Some packages need to be installed for these modules to execute properly, say galois, pickle collections, etc. We run
above modules on spyder 5.* using python 3.7 of tensorflow 2.X.



