# Short_BCH_Decoding_OSD
For the project focusing on parallel decoding of NMS for BCH codes of block length 63 and 127, refer to the weblink:
[Effective Application of Normalized Min-Sum Decoding for BCH Codes](https://arxiv.org/abs/2412.20828), the related subdirectories 
are Training_data_gen_63, Tesing_data_gen_63, BCH_63_training and BCH_63_testing in the main directory, 
while altering the string '63' of these subdirectories to '127' and replacing the arguments with these matched to BCH codes of length 127
will work without modifying code itself.

For the extened project to adopt hybrid rchitecture of NMS+DIA+OSD, holding the benefits of low-complexity low-latency,
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
above modules on spyder 5.2.2 using python 3.7 of tensorflow 2.X.



