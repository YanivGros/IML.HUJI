from unittest import TestCase

import numpy as np

from IMLearn.learners.regressors.polynomial_fitting import PolynomialFitting

class TestPolynomialFitting(TestCase):
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    p = PolynomialFitting(10)
    p._transform(x)
    print()

    pass
