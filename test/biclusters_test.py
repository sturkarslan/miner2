import sys
import unittest

import pandas as pd
import numpy as np
import logging
import json
import os
from collections import defaultdict

from miner2 import biclusters, preprocess


class BiclustersTest(unittest.TestCase):

    def compare_dicts(self, d1, d2):
        """compare 1-level deep dictionary"""
        ref_keys = sorted(d1.keys())
        keys = sorted(d2.keys())
        self.assertEquals(ref_keys, keys)

        for key in keys:
            ref_genes = sorted(d1[key])
            genes = sorted(d2[key])
            if len(ref_genes) != len(genes):
                print("MISMATCH KEY: '%s'" % key)
                print('REF GENES')
                print(ref_genes)
                print('GENES')
                print(genes)
            self.assertEquals(ref_genes, genes)

    def test_make_overexpressed_members(self):
        with open('testdata/ref_regulon_modules-001.json') as infile:
            regulon_modules = json.load(infile)
        with open('testdata/ref_ovx_membs-001.json') as infile:
            ref_ovx_membs = json.load(infile)
        bkgd = pd.read_csv('testdata/bkgd-001.csv', index_col=0, header=0)

        ovx_membs = biclusters.make_membership_dictionary(regulon_modules,
                                                          bkgd, label=2, p=0.05)
        #with open('testdata/ref_ovx_membs-001.json', 'w') as outfile:
        #    json.dump(ovx_membs, outfile)
        self.compare_dicts(ref_ovx_membs, ovx_membs)

    def test_make_underexpressed_members(self):
        with open('testdata/ref_regulon_modules-001.json') as infile:
            regulon_modules = json.load(infile)
        bkgd = pd.read_csv('testdata/bkgd-001.csv', index_col=0, header=0)
        with open('testdata/ref_undx_membs-001.json') as infile:
            ref_undx_membs = json.load(infile)

        undx_membs = biclusters.make_membership_dictionary(regulon_modules,
                                                           bkgd, label=0, p=0.05)

        #with open('testdata/ref_undx_membs-001.json', 'w') as outfile:
        #    json.dump(undx_membs, outfile)
        self.compare_dicts(ref_undx_membs, undx_membs)


    def test_membership_to_incidence(self):
        exp_data, conv_table = preprocess.main('testdata/ref_exp-000.csv',
                                               'testdata/identifier_mappings.txt')
        with open('testdata/ref_ovx_membs-001.json') as infile:
            ovx_membs = json.load(infile)
        ref_ovx_matrix = pd.read_csv('testdata/ref_ovx_matrix-001.csv', index_col=0, header=0)
        ovx_matrix = biclusters.membership_to_incidence(ovx_membs, exp_data)
        #ovx_matrix.to_csv('testdata/ref_ovx_matrix-001.csv')
        self.assertTrue(np.isclose(ref_ovx_matrix, ovx_matrix).all())


if __name__ == '__main__':
    SUITE = []
    LOG_FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S \t')
    SUITE.append(unittest.TestLoader().loadTestsFromTestCase(BiclustersTest))
    if len(sys.argv) > 1 and sys.argv[1] == 'xml':
      xmlrunner.XMLTestRunner(output='test-reports').run(unittest.TestSuite(SUITE))
    else:
      unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(SUITE))
