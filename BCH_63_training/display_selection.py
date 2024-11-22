# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 00:34:56 2021

@author: Administrator
"""
import os
import globalmap as GL
def display_selection():
    print("My piD: " + str(os.getpid()))
    if GL.get_map('selected_decoder_type') == 'SPA':
        print("Using Sum-Product Algorithm")
    elif GL.get_map('selected_decoder_type') == ('NNMS' or 'SNNMS'):
        print("Using Min-Sum algorithm")
    