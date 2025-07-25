import unittest.runner

if not hasattr(unittest.runner, '_TextTestResult'):
    unittest.runner._TextTestResult = unittest.runner.TextTestResult
