#!/usr/bin/env python3
import sys
import unittest

import pandas as pd
import numpy as np
from miner2 import preprocess
from miner2 import coexpression

class CoexpressionTest(unittest.TestCase):

    def test_cluster(self):
        ref_exp = pd.read_csv('testdata/expected_pp_exp_data-002.csv', index_col=0, header=0)
        exp, conv_table = preprocess.main('testdata/exp_data-002.csv', 'testdata/conv_table-002.tsv')
        self.assertTrue(np.isclose(ref_exp, exp).all())  # just to make sure
        expected_clusters = []
        with open('testdata/expected_clusters-002.csv', 'r') as infile:
            for line in infile:
                expected_clusters.append(line.strip().split(','))
        clusters = coexpression.cluster(exp)
        #with open('testdata/expected_clusters-002.csv', 'w') as outfile:
        #    for cluster in clusters:
        #        outfile.write('%s\n' % (','.join(cluster)))
        self.assertEquals(len(expected_clusters), len(clusters))
        self.assertEquals(expected_clusters, clusters)


if __name__ == '__main__':
    SUITE = []
    SUITE.append(unittest.TestLoader().loadTestsFromTestCase(CoexpressionTest))
    if len(sys.argv) > 1 and sys.argv[1] == 'xml':
      xmlrunner.XMLTestRunner(output='test-reports').run(unittest.TestSuite(SUITE))
    else:
      unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(SUITE))
