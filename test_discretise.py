import unittest
import numpy as np
from discretise import entropy, information_gain, discretise

"""
Test class for discretise.py
"""

class Test_discretise(unittest.TestCase):
  print("testing..")
 
  def test_entropy(self):
    X = np.array([[2,0], [5,0], [3,0], [2,1], [1,0], [10,0], [7,1], [6,1], [4,0]])
    self.assertAlmostEqual(entropy(X), 0.918, places=3)    

  def test_infogain(self):
    X = np.array([[2,0], [5,0], [3,0], [2,1], [1,0], [10,0], [7,1], [6,1], [4,0]])
    bin1 = np.array([[2,0], [5,0], [3,0], [2,1], [1,0], [4,0]])
    bin2 = np.array([[6,1], [7,1], [10,0]])
    self.assertAlmostEqual(information_gain(X, bin1, bin2), 0.179, places=3)

  def test_discretise(self):
    X = np.array([[2,0], [5,0], [3,0], [2,1], [1,0], [10,0], [7,1], [6,1], [4,0]])
    print(discretise(X))
    
  if __name__ == '__main__':
    unittest.main()


